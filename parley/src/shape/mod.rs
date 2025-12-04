// Copyright 2021 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Text shaping implementation using `harfrust`for shaping
//! and `icu` for text analysis.

use core::ops::RangeInclusive;

use alloc::vec::Vec;

use super::layout::Layout;
use super::resolve::{RangedStyle, ResolveContext, Resolved};
use super::style::{Brush, FontFeature, FontVariation};
use crate::analysis::cluster::CharCluster;
use crate::analysis::{AnalysisDataSources, CharInfo};
use crate::icu_convert::script_to_harfrust;
use crate::inline_box::InlineBox;
use crate::lru_cache::LruCache;
use crate::util::nearly_eq;
use crate::{FontData, icu_convert};
use icu_locale_core::LanguageIdentifier;
use icu_properties::props::Script;

use fontique::{self, Query, QueryFamily, QueryFont};

pub(crate) mod cache;

pub(crate) struct ShapeContext {
    shape_data_cache: LruCache<cache::ShapeDataKey, harfrust::ShaperData>,
    shape_instance_cache: LruCache<cache::ShapeInstanceId, harfrust::ShaperInstance>,
    shape_plan_cache: LruCache<cache::ShapePlanId, harfrust::ShapePlan>,
    metrics_cache: LruCache<cache::MetricsCacheId, cache::CachedMetrics>,
    unicode_buffer: Option<harfrust::UnicodeBuffer>,
    features: Vec<harfrust::Feature>,
    char_cluster: CharCluster,
    // Allocation pools
    cluster_pool: cache::VecPool<crate::layout::data::ClusterData>,
    glyph_pool: cache::VecPool<crate::layout::Glyph>,
    tried_fonts_pool: cache::VecPool<SelectedFont>,
    hole_pool: cache::VecPool<crate::layout::data::Hole>,
}

impl Default for ShapeContext {
    fn default() -> Self {
        const MAX_CACHE_ENTRIES: usize = 16;
        const MAX_POOL_SIZE: usize = 4;
        Self {
            shape_data_cache: LruCache::new(MAX_CACHE_ENTRIES),
            shape_instance_cache: LruCache::new(MAX_CACHE_ENTRIES),
            shape_plan_cache: LruCache::new(MAX_CACHE_ENTRIES),
            metrics_cache: LruCache::new(MAX_CACHE_ENTRIES),
            unicode_buffer: Some(harfrust::UnicodeBuffer::new()),
            features: Vec::new(),
            char_cluster: CharCluster::default(),
            cluster_pool: cache::VecPool::new(MAX_POOL_SIZE),
            glyph_pool: cache::VecPool::new(MAX_POOL_SIZE),
            tried_fonts_pool: cache::VecPool::new(MAX_POOL_SIZE),
            hole_pool: cache::VecPool::new(MAX_POOL_SIZE),
        }
    }
}

struct Item {
    style_index: u16,
    size: f32,
    script: Script,
    level: u8,
    locale: Option<LanguageIdentifier>,
    variations: Resolved<FontVariation>,
    features: Resolved<FontFeature>,
    word_spacing: f32,
    letter_spacing: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn shape_text<'a, B: Brush>(
    rcx: &'a ResolveContext,
    mut fq: Query<'a>,
    styles: &'a [RangedStyle<B>],
    inline_boxes: &[InlineBox],
    infos: &[(CharInfo, u16)],
    scx: &mut ShapeContext,
    mut text: &str,
    layout: &mut Layout<B>,
    analysis_data_sources: &AnalysisDataSources,
) {
    // If we have both empty text and no inline boxes, shape with a fake space
    // to generate metrics that can be used to size a cursor.
    if text.is_empty() && inline_boxes.is_empty() {
        text = " ";
    }
    // Do nothing if there is no text or styles (there should always be a default style)
    if text.is_empty() || styles.is_empty() {
        // Process any remaining inline boxes whose index is greater than the length of the text
        for box_idx in 0..inline_boxes.len() {
            // Push the box to the list of items
            layout.data.push_inline_box(box_idx);
        }
        return;
    }

    // Setup mutable state for iteration
    let mut style = &styles[0].style;
    let mut item = Item {
        style_index: 0,
        size: style.font_size,
        level: infos[0].0.bidi_embed_level,
        script: infos
            .iter()
            .map(|x| x.0.script)
            .find(|&script| real_script(script))
            .unwrap_or(Script::Latin),
        locale: style.locale.clone(),
        variations: style.font_variations,
        features: style.font_features,
        word_spacing: style.word_spacing,
        letter_spacing: style.letter_spacing,
    };

    let mut char_range = 0..0;
    let mut text_range = 0..0;

    let mut inline_box_iter = inline_boxes.iter().enumerate();
    let mut current_box = inline_box_iter.next();

    // Iterate over characters in the text
    for ((_, (byte_index, ch)), (info, style_index)) in text.char_indices().enumerate().zip(infos) {
        let mut break_run = false;
        let mut script = info.script;
        if !real_script(script) {
            script = item.script;
        }
        let level = info.bidi_embed_level;
        if item.style_index != *style_index {
            item.style_index = *style_index;
            style = &styles[*style_index as usize].style;
            if !nearly_eq(style.font_size, item.size)
                || style.locale != item.locale
                || style.font_variations != item.variations
                || style.font_features != item.features
                || !nearly_eq(style.letter_spacing, item.letter_spacing)
                || !nearly_eq(style.word_spacing, item.word_spacing)
            {
                break_run = true;
            }
        }

        if level != item.level || script != item.script {
            break_run = true;
        }

        // Check if there is an inline box at this index
        // Note:
        //   - We loop because there may be multiple boxes at this index
        //   - We do this *before* processing the text run because we need to know whether we should
        //     break the run due to the presence of an inline box.
        let mut deferred_boxes: Option<RangeInclusive<usize>> = None;
        while let Some((box_idx, inline_box)) = current_box {
            if inline_box.index == byte_index {
                break_run = true;
                if let Some(boxes) = &mut deferred_boxes {
                    deferred_boxes = Some((*boxes.start())..=box_idx);
                } else {
                    deferred_boxes = Some(box_idx..=box_idx);
                };
                // Update the current box to the next box
                current_box = inline_box_iter.next();
            } else {
                break;
            }
        }

        if break_run && !text_range.is_empty() {
            shape_item(
                &mut fq,
                rcx,
                styles,
                &item,
                scx,
                text,
                &text_range,
                &char_range,
                infos,
                layout,
                analysis_data_sources,
            );
            item.size = style.font_size;
            item.level = level;
            item.script = script;
            item.locale = style.locale.clone();
            item.variations = style.font_variations;
            item.features = style.font_features;
            text_range.start = text_range.end;
            char_range.start = char_range.end;
        }

        if let Some(deferred_boxes) = deferred_boxes {
            for box_idx in deferred_boxes {
                layout.data.push_inline_box(box_idx);
            }
        }

        text_range.end += ch.len_utf8();
        char_range.end += 1;
    }

    if !text_range.is_empty() {
        shape_item(
            &mut fq,
            rcx,
            styles,
            &item,
            scx,
            text,
            &text_range,
            &char_range,
            infos,
            layout,
            analysis_data_sources,
        );
    }

    // Process any remaining inline boxes whose index is greater than the length of the text
    if let Some((box_idx, _inline_box)) = current_box {
        layout.data.push_inline_box(box_idx);
    }
    for (box_idx, _inline_box) in inline_box_iter {
        layout.data.push_inline_box(box_idx);
    }
}

// Rebuilds the provided `char_cluster` in-place using the existing allocation
// for the given grapheme `segment_text`, consuming items from `item_infos_iter`.
fn fill_cluster_in_place(
    segment_text: &str,
    item_infos_iter: &mut core::slice::Iter<'_, (CharInfo, u16)>,
    code_unit_offset_in_string: &mut usize,
    char_cluster: &mut CharCluster,
) {
    // Reset cluster but keep allocation
    // TODO: Remove force normalise from info.
    char_cluster.clear();

    let mut cluster_style_index = u16::MAX;
    let mut is_emoji_or_pictograph = false;
    let start = *code_unit_offset_in_string as u32;

    for ((_, ch), (info, style_index)) in segment_text.char_indices().zip(item_infos_iter.by_ref())
    {
        cluster_style_index = *style_index;
        // TODO - make emoji detection more complete, as per (except using composite Trie tables as
        //  much as possible:
        //  https://github.com/conor-93/parley/blob/4637d826732a1a82bbb3c904c7f47a16a21cceec/parley/src/shape/mod.rs#L221-L269
        is_emoji_or_pictograph |= info.is_emoji_or_pictograph();
        *code_unit_offset_in_string += ch.len_utf8();
    }

    // Finalize cluster metadata
    let end = *code_unit_offset_in_string as u32;
    char_cluster.is_emoji = is_emoji_or_pictograph;
    char_cluster.style_index = cluster_style_index;
    char_cluster.start = start;
    char_cluster.end = end;
}

fn shape_item<'a, B: Brush>(
    fq: &mut Query<'a>,
    rcx: &'a ResolveContext,
    styles: &'a [RangedStyle<B>],
    item: &Item,
    scx: &mut ShapeContext,
    text: &str,
    text_range: &core::ops::Range<usize>,
    char_range: &core::ops::Range<usize>,
    infos: &[(CharInfo, u16)],
    layout: &mut Layout<B>,
    analysis_data_sources: &AnalysisDataSources,
) {
    let item_text = &text[text_range.clone()];
    let item_infos = &infos[char_range.start..char_range.end];
    let first_style_index = item_infos[0].1;
    let fb_script = icu_convert::script_to_fontique(item.script, analysis_data_sources);
    let mut font_selector = FontSelector::new(
        fq,
        rcx,
        styles,
        first_style_index,
        fb_script,
        item.locale.clone(),
    );

    let grapheme_cluster_boundaries = analysis_data_sources
        .grapheme_segmenter()
        .segment_str(item_text);
    let mut item_infos_iter = item_infos.iter();
    let mut code_unit_offset_in_string = text_range.start;

    // Build an iterator of boundaries and consume the first segment to seed the loop
    let mut boundaries_iter = grapheme_cluster_boundaries.skip(1);
    let mut last_boundary = 0_usize;
    let Some(mut current_boundary) = boundaries_iter.next() else {
        return; // No clusters
    };

    fill_cluster_in_place(
        &item_text[last_boundary..current_boundary],
        &mut item_infos_iter,
        &mut code_unit_offset_in_string,
        &mut scx.char_cluster,
    );

    let mut current_font = font_selector.select_font(&mut scx.char_cluster);

    // Main segmentation loop - segment by font changes
    while let Some(font) = current_font.take() {
        // Collect all clusters for this font segment
        let cluster_range = scx.char_cluster.range();
        let segment_start_offset = cluster_range.start as usize - text_range.start;
        let mut segment_end_offset = cluster_range.end as usize - text_range.start;

        for next_boundary in boundaries_iter.by_ref() {
            // Build next cluster in-place
            last_boundary = current_boundary;
            current_boundary = next_boundary;
            fill_cluster_in_place(
                &item_text[last_boundary..current_boundary],
                &mut item_infos_iter,
                &mut code_unit_offset_in_string,
                &mut scx.char_cluster,
            );

            if let Some(next_font) = font_selector.select_font(&mut scx.char_cluster) {
                if next_font != font {
                    current_font = Some(next_font);
                    break;
                } else {
                    // Same font - add to current segment
                    segment_end_offset = scx.char_cluster.range().end as usize - text_range.start;
                }
            } else {
                // No font determined, continue to next cluster
                continue;
            }
        }

        // Shape this font segment
        let segment_text = &item_text[segment_start_offset..segment_end_offset];
        let char_start = char_range.start + item_text[..segment_start_offset].chars().count();
        let segment_char_start = char_start - char_range.start;
        let segment_char_count = segment_text.chars().count();
        let segment_infos =
            &item_infos[segment_char_start..(segment_char_start + segment_char_count)];

        // Shape with the selected font into ProcessedRun (without pushing to layout yet)
        let segment_abs_text_range =
            (text_range.start + segment_start_offset)..(text_range.start + segment_end_offset);
        let mut processed_run = shape_to_processed_run(
            rcx,
            item,
            scx,
            layout,
            &font,
            fb_script,
            segment_text,
            segment_infos,
            segment_abs_text_range.clone(),
        );

        // Analyze for holes and push, using fallback fonts for any holes
        let mut tried_fonts = scx.tried_fonts_pool.acquire();
        let mut holes = scx.hole_pool.acquire();
        push_run_with_fallback(
            rcx,
            item,
            scx,
            layout,
            &mut processed_run,
            &font, // primary_font - stays constant
            &font, // current_font - same as primary for initial call
            fb_script,
            segment_text,
            segment_infos,
            segment_abs_text_range,
            analysis_data_sources,
            &mut font_selector,
            &mut tried_fonts,
            &mut holes,
            true, // Reset tried_fonts for each sibling hole
        );
        scx.hole_pool.release(holes);
        scx.tried_fonts_pool.release(tried_fonts);
    }
}

/// Analyze a ProcessedRun for holes and push it to layout, using fallback fonts for holes.
///
/// This is the core logic shared between initial shaping and fallback shaping:
/// 1. Analyze the run for holes (glyph_id == 0)
/// 2. If no holes, push the entire run
/// 3. If holes, push shaped segments and recursively handle holes with fallback fonts
///
/// reset_per_hole:
///     If true, reset tried_fonts for each hole (for sibling holes at top level).
///     If false, accumulate tried_fonts across sub-holes (for recursive calls).
#[allow(clippy::too_many_arguments)]
fn push_run_with_fallback<'a, B: Brush>(
    rcx: &'a ResolveContext,
    item: &Item,
    scx: &mut ShapeContext,
    layout: &mut Layout<B>,
    processed: &mut crate::layout::data::ProcessedRun,
    primary_font: &SelectedFont,
    current_font: &SelectedFont,
    fb_script: fontique::Script,
    segment_text: &str,
    segment_infos: &[(CharInfo, u16)],
    text_range: core::ops::Range<usize>,
    analysis_data_sources: &AnalysisDataSources,
    font_selector: &mut FontSelector<'a, '_, B>,
    tried_fonts: &mut Vec<SelectedFont>,
    holes: &mut Vec<crate::layout::data::Hole>,
    reset_per_hole: bool,
) {
    use crate::layout::data::ShapedSegment;

    let analysis = crate::layout::data::LayoutData::<B>::analyze_processed_run(
        processed,
        segment_text.len(),
        holes,
    );

    if !analysis.has_holes {
        // No holes - push the entire run
        layout.data.push_processed_run(processed);
    } else {
        // Process each segment
        for segment in analysis.segments {
            match segment {
                ShapedSegment::Shaped {
                    cluster_range,
                    text_range: seg_text_range,
                } => {
                    // Push shaped portion
                    let abs_range = (text_range.start + seg_text_range.start)
                        ..(text_range.start + seg_text_range.end);
                    layout
                        .data
                        .push_processed_segment(processed, cluster_range, abs_range);
                }
                ShapedSegment::Hole(hole) => {
                    // Get hole text and infos
                    let hole_text = &segment_text[hole.text_range.clone()];
                    let hole_infos = &segment_infos[hole.char_range.clone()];

                    if hole_text.is_empty() || hole_infos.is_empty() {
                        continue;
                    }

                    let abs_hole_range = (text_range.start + hole.text_range.start)
                        ..(text_range.start + hole.text_range.end);

                    // Reset tried_fonts for each sibling hole at top level
                    if reset_per_hole {
                        tried_fonts.clear();
                        tried_fonts.push(current_font.clone());
                    }

                    // Configure font selector for this hole's style
                    let seg_boundaries = analysis_data_sources
                        .grapheme_segmenter()
                        .segment_str(hole_text);
                    let mut seg_iter = seg_boundaries.skip(1);
                    let first_seg_boundary = seg_iter.next().unwrap_or(hole_text.len());

                    let mut temp_infos_iter = hole_infos.iter();
                    let mut temp_offset = abs_hole_range.start;
                    fill_cluster_in_place(
                        &hole_text[0..first_seg_boundary],
                        &mut temp_infos_iter,
                        &mut temp_offset,
                        &mut scx.char_cluster,
                    );

                    // Call select_font to configure query
                    let _ = font_selector.select_font(&mut scx.char_cluster);

                    // Find next fallback font
                    let fallback = font_selector.select_next_font(tried_fonts);

                    match fallback {
                        Some(fallback_font) => {
                            // Shape with fallback font
                            let mut fallback_processed = shape_to_processed_run(
                                rcx,
                                item,
                                scx,
                                layout,
                                &fallback_font,
                                fb_script,
                                hole_text,
                                hole_infos,
                                abs_hole_range.clone(),
                            );

                            tried_fonts.push(fallback_font.clone());

                            // Recursively process sub-holes
                            push_run_with_fallback(
                                rcx,
                                item,
                                scx,
                                layout,
                                &mut fallback_processed,
                                primary_font,
                                &fallback_font,
                                fb_script,
                                hole_text,
                                hole_infos,
                                abs_hole_range.clone(),
                                analysis_data_sources,
                                font_selector,
                                tried_fonts,
                                holes,
                                false,
                            );

                            // Return fallback vecs to pool
                            let (clusters, glyphs) = fallback_processed.take_vecs();
                            scx.cluster_pool.release(clusters);
                            scx.glyph_pool.release(glyphs);
                        }
                        None => {
                            // No more fallbacks - shape with primary font for .notdef glyphs
                            let mut notdef_processed = shape_to_processed_run(
                                rcx,
                                item,
                                scx,
                                layout,
                                primary_font,
                                fb_script,
                                hole_text,
                                hole_infos,
                                abs_hole_range,
                            );
                            layout.data.push_processed_run(&notdef_processed);
                            let (clusters, glyphs) = notdef_processed.take_vecs();
                            scx.cluster_pool.release(clusters);
                            scx.glyph_pool.release(glyphs);
                        }
                    }
                }
            }
        }
    }

    // Return main run's vecs to pool
    let (clusters, glyphs) = processed.take_vecs();
    scx.cluster_pool.release(clusters);
    scx.glyph_pool.release(glyphs);
}

/// Shape a text segment with a specific font into a ProcessedRun (without pushing to layout).
/// This allows analyzing for holes before deciding how to push.
#[allow(clippy::too_many_arguments)]
fn shape_to_processed_run<B: Brush>(
    rcx: &ResolveContext,
    item: &Item,
    scx: &mut ShapeContext,
    layout: &mut Layout<B>,
    font: &SelectedFont,
    fb_script: fontique::Script,
    segment_text: &str,
    segment_infos: &[(CharInfo, u16)],
    text_range: core::ops::Range<usize>,
) -> crate::layout::data::ProcessedRun {
    let font_ref = harfrust::FontRef::from_index(font.font.blob.as_ref(), font.font.index).unwrap();

    let direction = if item.level & 1 != 0 {
        harfrust::Direction::RightToLeft
    } else {
        harfrust::Direction::LeftToRight
    };
    let hb_script = script_to_harfrust(fb_script);
    let language = item
        .locale
        .as_ref()
        .and_then(|lang| lang.language.as_str().parse::<harfrust::Language>().ok());

    // Acquire buffer from pool first (before cache borrows)
    let mut buffer = std::mem::take(&mut scx.unicode_buffer).unwrap();
    buffer.clear();
    buffer.reserve(segment_text.len());

    for (i, ch) in segment_text.chars().enumerate() {
        buffer.add(ch, i as u32);
    }
    buffer.set_direction(direction);
    buffer.set_script(hb_script);
    if let Some(ref lang) = language {
        buffer.set_language(lang.clone());
    }

    // Build features list
    scx.features.clear();
    for feature in rcx.features(item.features).unwrap_or(&[]) {
        scx.features.push(harfrust::Feature::new(
            harfrust::Tag::from_u32(feature.tag),
            feature.value as u32,
            ..,
        ));
    }

    // Build cache keys before borrowing caches
    let data_key = cache::ShapeDataKey::new(font.font.blob.id(), font.font.index);
    let instance_key = cache::ShapeInstanceKey::new(
        font.font.blob.id(),
        font.font.index,
        &font.font.synthesis,
        rcx.variations(item.variations),
    );
    let plan_key = cache::ShapePlanKey::new(
        font.font.blob.id(),
        font.font.index,
        &font.font.synthesis,
        direction,
        hb_script,
        language.clone(),
        &scx.features,
        rcx.variations(item.variations),
    );

    // Access caches and shape
    let (glyph_buffer, coords) = {
        let shaper_data = scx
            .shape_data_cache
            .entry(data_key, || harfrust::ShaperData::new(&font_ref));
        let instance = scx.shape_instance_cache.entry(instance_key, || {
            harfrust::ShaperInstance::from_variations(
                &font_ref,
                variations_iter(&font.font.synthesis, rcx.variations(item.variations)),
            )
        });

        let harf_shaper = shaper_data
            .shaper(&font_ref)
            .instance(Some(instance))
            .point_size(Some(item.size))
            .build();

        let shaper_plan = scx.shape_plan_cache.entry(plan_key, || {
            harfrust::ShapePlan::new(
                &harf_shaper,
                direction,
                Some(hb_script),
                language.as_ref(),
                &scx.features,
            )
        });

        let glyph_buffer = harf_shaper.shape_with_plan(shaper_plan, buffer, &scx.features);
        let coords: Vec<_> = harf_shaper.coords().to_vec();
        (glyph_buffer, coords)
    };

    // Get cached metrics or compute them (no allocation for lookup)
    let metrics_key =
        cache::MetricsCacheKey::new(font.font.blob.id(), font.font.index, item.size, &coords);
    let cached_metrics = scx.metrics_cache.entry(metrics_key, || {
        let skrifa_font = skrifa::FontRef::from_index(font.font.blob.as_ref(), font.font.index)
            .expect("failed to create skrifa font ref");
        // Convert coords to skrifa's expected format
        let skrifa_coords: Vec<skrifa::instance::NormalizedCoord> = coords
            .iter()
            .map(|c| skrifa::instance::NormalizedCoord::from_bits(c.to_bits()))
            .collect();
        let metrics = skrifa::metrics::Metrics::new(
            &skrifa_font,
            skrifa::prelude::Size::new(item.size),
            skrifa_coords.as_slice(),
        );
        let units_per_em = metrics.units_per_em as f32;
        let (underline_offset, underline_size) = if let Some(underline) = metrics.underline {
            (underline.offset, underline.thickness)
        } else {
            let default = units_per_em / 18.0;
            (default, default)
        };
        let (strikethrough_offset, strikethrough_size) = if let Some(strikeout) = metrics.strikeout
        {
            (strikeout.offset, strikeout.thickness)
        } else {
            (metrics.ascent / 2.0, units_per_em / 18.0)
        };
        cache::CachedMetrics {
            units_per_em,
            ascent: metrics.ascent,
            descent: -metrics.descent,
            leading: metrics.leading,
            underline_offset,
            underline_size,
            strikethrough_offset,
            strikethrough_size,
        }
    });

    // Convert coords to i16 for storage (owned, passed directly to avoid clone)
    let coords_i16: Vec<i16> = coords.iter().map(|c| c.to_bits()).collect();

    // Acquire vectors from pools and reserve capacity
    let mut clusters = scx.cluster_pool.acquire();
    clusters.reserve(segment_infos.len());
    let mut glyphs = scx.glyph_pool.acquire();
    // Estimate ~1.5 glyphs per character (handles ligatures, complex scripts)
    glyphs.reserve(segment_infos.len() + segment_infos.len() / 2);

    // Process to temp storage (don't push yet)
    let processed = layout.data.process_run_to_temp(
        FontData::new(font.font.blob.clone(), font.font.index),
        item.size,
        font.font.synthesis.clone(),
        &glyph_buffer,
        item.level,
        item.style_index,
        item.word_spacing,
        item.letter_spacing,
        segment_text,
        segment_infos,
        text_range,
        coords_i16, // Pass by ownership
        cached_metrics,
        clusters,
        glyphs,
    );

    // Return buffer to context
    scx.unicode_buffer = Some(glyph_buffer.clear());

    processed
}

fn real_script(script: Script) -> bool {
    script != Script::Common && script != Script::Unknown && script != Script::Inherited
}

fn variations_iter<'a>(
    synthesis: &'a fontique::Synthesis,
    item: Option<&'a [FontVariation]>,
) -> impl Iterator<Item = harfrust::Variation> + 'a {
    synthesis
        .variation_settings()
        .iter()
        .map(|(tag, value)| harfrust::Variation {
            tag: *tag,
            value: *value,
        })
        .chain(
            item.unwrap_or(&[])
                .iter()
                .map(|variation| harfrust::Variation {
                    tag: harfrust::Tag::from_u32(variation.tag),
                    value: variation.value,
                }),
        )
}

struct FontSelector<'a, 'b, B: Brush> {
    query: &'b mut Query<'a>,
    fonts_id: Option<usize>,
    rcx: &'a ResolveContext,
    styles: &'a [RangedStyle<B>],
    style_index: u16,
    attrs: fontique::Attributes,
    variations: &'a [FontVariation],
    features: &'a [FontFeature],
}

impl<'a, 'b, B: Brush> FontSelector<'a, 'b, B> {
    fn new(
        query: &'b mut Query<'a>,
        rcx: &'a ResolveContext,
        styles: &'a [RangedStyle<B>],
        style_index: u16,
        fb_script: fontique::Script,
        locale: Option<LanguageIdentifier>,
    ) -> Self {
        let style = &styles[style_index as usize].style;
        let fonts_id = style.font_stack.id();
        let fonts = rcx.stack(style.font_stack).unwrap_or(&[]);
        let attrs = fontique::Attributes {
            width: style.font_width,
            weight: style.font_weight,
            style: style.font_style,
        };
        let variations = rcx.variations(style.font_variations).unwrap_or(&[]);
        let features = rcx.features(style.font_features).unwrap_or(&[]);
        query.set_families(fonts.iter().copied());

        let fb_language = locale.and_then(icu_convert::locale_to_fontique);
        query.set_fallbacks(fontique::FallbackKey::new(fb_script, fb_language.as_ref()));
        query.set_attributes(attrs);

        Self {
            query,
            fonts_id: Some(fonts_id),
            rcx,
            styles,
            style_index,
            attrs,
            variations,
            features,
        }
    }

    /// Select a font for the given cluster.
    ///
    /// Does not check character coverage.
    fn select_font(&mut self, cluster: &mut CharCluster) -> Option<SelectedFont> {
        let style_index = cluster.style_index();
        let is_emoji = cluster.is_emoji;
        if style_index != self.style_index || is_emoji || self.fonts_id.is_none() {
            self.style_index = style_index;
            let style = &self.styles[style_index as usize].style;

            let fonts_id = style.font_stack.id();
            let fonts = self.rcx.stack(style.font_stack).unwrap_or(&[]);
            let fonts = fonts.iter().copied().map(QueryFamily::Id);
            if is_emoji {
                use core::iter::once;
                let emoji_family = QueryFamily::Generic(fontique::GenericFamily::Emoji);
                self.query.set_families(fonts.chain(once(emoji_family)));
                self.fonts_id = None;
            } else if self.fonts_id != Some(fonts_id) {
                self.query.set_families(fonts);
                self.fonts_id = Some(fonts_id);
            }

            let attrs = fontique::Attributes {
                width: style.font_width,
                weight: style.font_weight,
                style: style.font_style,
            };
            if self.attrs != attrs {
                self.query.set_attributes(attrs);
                self.attrs = attrs;
            }
            self.variations = self.rcx.variations(style.font_variations).unwrap_or(&[]);
            self.features = self.rcx.features(style.font_features).unwrap_or(&[]);
        }

        // Just return the first matching font - no coverage checking.
        let mut selected_font = None;
        self.query.matches_with(|font| {
            selected_font = Some(font.into());
            fontique::QueryStatus::Stop
        });
        selected_font
    }

    /// Select the next font in the fallback chain, skipping any fonts we've already tried.
    fn select_next_font(&mut self, skip_fonts: &[SelectedFont]) -> Option<SelectedFont> {
        let mut selected_font = None;
        self.query.matches_with(|font| {
            let candidate: SelectedFont = font.into();
            if skip_fonts.iter().any(|skip| *skip == candidate) {
                return fontique::QueryStatus::Continue;
            }
            // Return the first font we haven't tried yet
            selected_font = Some(candidate);
            fontique::QueryStatus::Stop
        });
        selected_font
    }
}

#[derive(Clone)]
struct SelectedFont {
    font: QueryFont,
}

impl From<&QueryFont> for SelectedFont {
    fn from(font: &QueryFont) -> Self {
        Self { font: font.clone() }
    }
}

impl PartialEq for SelectedFont {
    fn eq(&self, other: &Self) -> bool {
        self.font.family == other.font.family && self.font.synthesis == other.font.synthesis
    }
}
