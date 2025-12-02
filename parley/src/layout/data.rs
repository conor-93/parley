// Copyright 2021 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::inline_box::InlineBox;
use crate::layout::{ContentWidths, Glyph, LineMetrics, RunMetrics, Style};
use crate::style::Brush;
use crate::util::nearly_zero;
use crate::{FontData, LineHeight, OverflowWrap, TextWrapMode};
use core::ops::Range;

use alloc::vec::Vec;

use crate::analysis::cluster::Whitespace;
use crate::analysis::{Boundary, CharInfo};
#[cfg(feature = "libm")]
#[allow(unused_imports)]
use core_maths::CoreFloat;

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct ClusterData {
    pub(crate) info: ClusterInfo,
    /// Cluster flags (see impl methods for details).
    pub(crate) flags: u16,
    /// Style index for this cluster.
    pub(crate) style_index: u16,
    /// Number of glyphs in this cluster (0xFF = single glyph stored inline)
    pub(crate) glyph_len: u8,
    /// Number of text bytes in this cluster
    pub(crate) text_len: u8,
    /// If `glyph_len == 0xFF`, then `glyph_offset` is a glyph identifier,
    /// otherwise, it's an offset into the glyph array with the base
    /// taken from the owning run.
    pub(crate) glyph_offset: u32,
    /// Offset into the text for this cluster
    pub(crate) text_offset: u16,
    /// Advance width for this cluster
    pub(crate) advance: f32,
}

impl ClusterData {
    pub(crate) const LIGATURE_START: u16 = 1;
    pub(crate) const LIGATURE_COMPONENT: u16 = 2;

    #[inline(always)]
    pub(crate) fn is_ligature_start(self) -> bool {
        self.flags & Self::LIGATURE_START != 0
    }

    #[inline(always)]
    pub(crate) fn is_ligature_component(self) -> bool {
        self.flags & Self::LIGATURE_COMPONENT != 0
    }

    #[inline(always)]
    pub(crate) fn text_range(self, run: &RunData) -> Range<usize> {
        let start = run.text_range.start + self.text_offset as usize;
        start..start + self.text_len as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct ClusterInfo {
    boundary: Boundary,
    source_char: char,
}

impl ClusterInfo {
    pub(crate) fn new(boundary: Boundary, source_char: char) -> Self {
        Self {
            boundary,
            source_char,
        }
    }

    // Returns the boundary type of the cluster.
    pub(crate) fn boundary(self) -> Boundary {
        self.boundary
    }

    // Returns the whitespace type of the cluster.
    pub(crate) fn whitespace(self) -> Whitespace {
        to_whitespace(self.source_char)
    }

    /// Returns if the cluster is a line boundary.
    pub(crate) fn is_boundary(self) -> bool {
        self.boundary != Boundary::None
    }

    /// Returns if the cluster is an emoji.
    pub(crate) fn is_emoji(self) -> bool {
        // TODO: Defer to ICU4X properties (see: https://docs.rs/icu/latest/icu/properties/props/struct.Emoji.html).
        matches!(self.source_char as u32, 0x1F600..=0x1F64F | 0x1F300..=0x1F5FF | 0x1F680..=0x1F6FF | 0x2600..=0x26FF | 0x2700..=0x27BF)
    }

    /// Returns if the cluster is any whitespace.
    pub(crate) fn is_whitespace(self) -> bool {
        self.source_char.is_whitespace()
    }

    #[cfg(test)]
    pub(crate) fn source_char(self) -> char {
        self.source_char
    }
}

fn to_whitespace(c: char) -> Whitespace {
    match c {
        ' ' => Whitespace::Space,
        '\t' => Whitespace::Tab,
        '\n' => Whitespace::Newline,
        '\r' => Whitespace::Newline,
        '\u{00A0}' => Whitespace::NoBreakSpace,
        _ => Whitespace::None,
    }
}

/// `HarfRust`-based run data
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RunData {
    /// Index of the font for the run.
    pub(crate) font_index: usize,
    /// Font size.
    pub(crate) font_size: f32,
    /// Synthesis for rendering (contains variation settings)
    pub(crate) synthesis: fontique::Synthesis,
    /// Range of normalized coordinates in the layout data.
    pub(crate) coords_range: Range<usize>,
    /// Range of the source text.
    pub(crate) text_range: Range<usize>,
    /// Bidi level for the run.
    pub(crate) bidi_level: u8,
    /// Range of clusters.
    pub(crate) cluster_range: Range<usize>,
    /// Base for glyph indices.
    pub(crate) glyph_start: usize,
    /// Metrics for the run.
    pub(crate) metrics: RunMetrics,
    /// Additional word spacing.
    pub(crate) word_spacing: f32,
    /// Additional letter spacing.
    pub(crate) letter_spacing: f32,
    /// Total advance of the run.
    pub(crate) advance: f32,
}

#[derive(Copy, Clone, Default, PartialEq, Debug)]
pub enum BreakReason {
    #[default]
    None,
    Regular,
    Explicit,
    Emergency,
}

#[derive(Clone, Default, Debug, PartialEq)]
pub(crate) struct LineData {
    /// Range of the source text.
    pub(crate) text_range: Range<usize>,
    /// Range of line items.
    pub(crate) item_range: Range<usize>,
    /// Metrics for the line.
    pub(crate) metrics: LineMetrics,
    /// The cause of the line break.
    pub(crate) break_reason: BreakReason,
    /// Maximum advance for the line.
    pub(crate) max_advance: f32,
    /// Number of justified clusters on the line.
    pub(crate) num_spaces: usize,
}

impl LineData {
    pub(crate) fn size(&self) -> f32 {
        self.metrics.ascent + self.metrics.descent + self.metrics.leading
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LineItemData {
    /// Whether the item is a run or an inline box
    pub(crate) kind: LayoutItemKind,
    /// The index of the run or inline box in the runs or `inline_boxes` vec
    pub(crate) index: usize,
    /// Bidi level for the item (used for reordering)
    pub(crate) bidi_level: u8,
    /// Advance (size in direction of text flow) for the run.
    pub(crate) advance: f32,

    // Fields that only apply to text runs (Ignored for boxes)
    // TODO: factor this out?
    /// True if the run is composed entirely of whitespace.
    pub(crate) is_whitespace: bool,
    /// True if the run ends in whitespace.
    pub(crate) has_trailing_whitespace: bool,
    /// Range of the source text.
    pub(crate) text_range: Range<usize>,
    /// Range of clusters.
    pub(crate) cluster_range: Range<usize>,
}

impl LineItemData {
    pub(crate) fn is_text_run(&self) -> bool {
        self.kind == LayoutItemKind::TextRun
    }

    #[inline(always)]
    pub(crate) fn is_rtl(&self) -> bool {
        self.bidi_level & 1 != 0
    }

    /// If the item is a text run
    ///   - Determine if it consists entirely of whitespace (`is_whitespace` property)
    ///   - Determine if it has trailing whitespace (`has_trailing_whitespace` property)
    pub(crate) fn compute_whitespace_properties<B: Brush>(&mut self, layout_data: &LayoutData<B>) {
        // Skip items which are not text runs
        if self.kind != LayoutItemKind::TextRun {
            return;
        }

        self.is_whitespace = true;
        if self.is_rtl() {
            // RTL runs check for "trailing" whitespace at the front.
            for cluster in layout_data.clusters[self.cluster_range.clone()].iter() {
                if cluster.info.is_whitespace() {
                    self.has_trailing_whitespace = true;
                } else {
                    self.is_whitespace = false;
                    break;
                }
            }
        } else {
            for cluster in layout_data.clusters[self.cluster_range.clone()]
                .iter()
                .rev()
            {
                if cluster.info.is_whitespace() {
                    self.has_trailing_whitespace = true;
                } else {
                    self.is_whitespace = false;
                    break;
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayoutItemKind {
    TextRun,
    InlineBox,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayoutItem {
    /// Whether the item is a run or an inline box
    pub(crate) kind: LayoutItemKind,
    /// The index of the run or inline box in the runs or `inline_boxes` vec
    pub(crate) index: usize,
    /// Bidi level for the item (used for reordering)
    pub(crate) bidi_level: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayoutData<B: Brush> {
    pub(crate) scale: f32,
    pub(crate) quantize: bool,
    pub(crate) base_level: u8,
    pub(crate) text_len: usize,
    pub(crate) width: f32,
    pub(crate) full_width: f32,
    pub(crate) height: f32,
    pub(crate) fonts: Vec<FontData>,
    pub(crate) coords: Vec<i16>,

    // Input (/ output of style resolution)
    pub(crate) styles: Vec<Style<B>>,
    pub(crate) inline_boxes: Vec<InlineBox>,

    // Output of shaping
    pub(crate) runs: Vec<RunData>,
    pub(crate) items: Vec<LayoutItem>,
    pub(crate) clusters: Vec<ClusterData>,
    pub(crate) glyphs: Vec<Glyph>,

    // Output of line breaking
    pub(crate) lines: Vec<LineData>,
    pub(crate) line_items: Vec<LineItemData>,

    // Output of alignment
    /// Whether the layout is aligned with [`crate::Alignment::Justify`].
    pub(crate) is_aligned_justified: bool,
    /// The width the layout was aligned to.
    pub(crate) alignment_width: f32,
}

/// Represents a contiguous range of unshaped clusters (glyphs with id 0) that need
/// to be reshaped with a fallback font.
#[derive(Clone, Debug)]
pub(crate) struct Hole {
    /// Char index range within the segment's char_infos slice.
    /// Used to extract the CharInfo slice for reshaping.
    pub char_range: Range<usize>,
    /// Text byte range within the segment text.
    /// Used to extract the substring for reshaping.
    pub text_range: Range<usize>,
}

/// A segment of shaped text, either successfully shaped or a hole needing fallback.
#[derive(Clone, Debug)]
pub(crate) enum ShapedSegment {
    /// Successfully shaped - glyphs are valid
    Shaped {
        /// Cluster index range (0-based within this shaping result)
        cluster_range: Range<usize>,
        /// Text byte range within the shaped text
        text_range: Range<usize>,
    },
    /// A hole - glyphs have glyph_id == 0, needs fallback font
    Hole(Hole),
}

/// Result of analyzing a glyph buffer for holes.
/// Contains alternating shaped and hole segments in text order.
#[derive(Clone, Debug)]
pub(crate) struct ShapedRunAnalysis {
    /// Segments in logical (text) order
    pub segments: Vec<ShapedSegment>,
    /// Whether any holes were found
    pub has_holes: bool,
}

/// Temporarily holds processed cluster/glyph data before deciding whether to push to layout.
/// This allows us to analyze for holes and then either push entirely or split by segments.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct ProcessedRun {
    /// Processed clusters (in logical order)
    pub clusters: Vec<ClusterData>,
    /// Processed glyphs
    pub glyphs: Vec<Glyph>,
    /// Total advance of the run
    pub advance: f32,
    /// Run metrics
    pub metrics: RunMetrics,
    /// Font data for this run
    pub font_index: usize,
    pub font_size: f32,
    pub synthesis: fontique::Synthesis,
    pub coords: Vec<i16>,
    /// Text info
    pub bidi_level: u8,
    pub style_index: u16,
    pub word_spacing: f32,
    pub letter_spacing: f32,
    pub text_range: Range<usize>,
}

impl ProcessedRun {
    /// Take ownership of the clusters and glyphs vectors for returning to pool.
    /// Returns (clusters, glyphs) vectors.
    pub(crate) fn take_vecs(&mut self) -> (Vec<ClusterData>, Vec<Glyph>) {
        (
            core::mem::take(&mut self.clusters),
            core::mem::take(&mut self.glyphs),
        )
    }
}

impl<B: Brush> Default for LayoutData<B> {
    fn default() -> Self {
        Self {
            scale: 1.,
            quantize: true,
            base_level: 0,
            text_len: 0,
            width: 0.,
            full_width: 0.,
            height: 0.,
            fonts: Vec::new(),
            coords: Vec::new(),
            styles: Vec::new(),
            inline_boxes: Vec::new(),
            runs: Vec::new(),
            items: Vec::new(),
            clusters: Vec::new(),
            glyphs: Vec::new(),
            lines: Vec::new(),
            line_items: Vec::new(),
            is_aligned_justified: false,
            alignment_width: 0.0,
        }
    }
}

impl<B: Brush> LayoutData<B> {
    pub(crate) fn clear(&mut self) {
        self.scale = 1.;
        self.quantize = true;
        self.base_level = 0;
        self.text_len = 0;
        self.width = 0.;
        self.full_width = 0.;
        self.height = 0.;
        self.fonts.clear();
        self.coords.clear();
        self.styles.clear();
        self.inline_boxes.clear();
        self.runs.clear();
        self.items.clear();
        self.clusters.clear();
        self.glyphs.clear();
        self.lines.clear();
        self.line_items.clear();
    }

    /// Push an inline box to the list of items
    pub(crate) fn push_inline_box(&mut self, index: usize) {
        // Give the box the same bidi level as the preceding text run
        // (or else default to 0 if there is not yet a text run)
        let bidi_level = self.runs.last().map(|r| r.bidi_level).unwrap_or(0);

        self.items.push(LayoutItem {
            kind: LayoutItemKind::InlineBox,
            index,
            bidi_level,
        });
    }

    // So the plan is:
    // 1. Shape the text without font selection
    // 2. If in layout analysis, we find holes (as determined by 0 glyph IDs), we:
    //      - If not empty, push the current state as a run
    //      - If there's more to process, push the current iterative state to a stack
    //          - Maybe just a reference to the original run + per-glyph-iterators
    //  3. Repeat from 1 until stack is empty, but where non-first iterations:
    //      - Keep track of total advance to push glyphs accordingly
    //
    //
    // Questions:
    // - How do we offset the glyphs that were positioned after some hole? Maybe it's as simple
    // as tracking the total advance, which we already do. We could add it to the run.

    // Invariants useful for shaper based itemisation:
    // - char_infos len is the total width of the clusters

    /// Push a shaped run to the layout.
    ///
    /// Returns a list of holes (ranges with glyph_id == 0) that need to be
    /// reshaped with fallback fonts. If the list is empty, the entire run
    /// was successfully shaped.
    /// Push a run directly from a glyph buffer (legacy method, kept for reference).
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(crate) fn push_run(
        &mut self,
        font: FontData,
        font_size: f32,
        synthesis: fontique::Synthesis,
        glyph_buffer: &harfrust::GlyphBuffer,
        bidi_level: u8,
        style_index: u16,
        word_spacing: f32,
        letter_spacing: f32,
        source_text: &str,
        char_infos: &[(CharInfo, u16)], // From text analysis
        text_range: Range<usize>,       // The text range this run covers
        coords: &[harfrust::NormalizedCoord],
    ) -> Vec<Hole> {
        let coords_start = self.coords.len();
        self.coords.extend(coords.iter().map(|c| c.to_bits()));
        let coords_end = self.coords.len();

        let font_index = self
            .fonts
            .iter()
            .position(|f| *f == font)
            .unwrap_or_else(|| {
                let index = self.fonts.len();
                self.fonts.push(font);
                index
            });

        let metrics = {
            let font = &self.fonts[font_index];
            let font_ref = skrifa::FontRef::from_index(font.data.as_ref(), font.index).unwrap();
            skrifa::metrics::Metrics::new(&font_ref, skrifa::prelude::Size::new(font_size), coords)
        };
        let units_per_em = metrics.units_per_em as f32;

        let metrics = {
            let (underline_offset, underline_size) = if let Some(underline) = metrics.underline {
                (underline.offset, underline.thickness)
            } else {
                // Default values from Harfbuzz: https://github.com/harfbuzz/harfbuzz/blob/00492ec7df0038f41f78d43d477c183e4e4c506e/src/hb-ot-metrics.cc#L334
                let default = units_per_em / 18.0;
                (default, default)
            };
            let (strikethrough_offset, strikethrough_size) =
                if let Some(strikeout) = metrics.strikeout {
                    (strikeout.offset, strikeout.thickness)
                } else {
                    // Default values from HarfBuzz: https://github.com/harfbuzz/harfbuzz/blob/00492ec7df0038f41f78d43d477c183e4e4c506e/src/hb-ot-metrics.cc#L334-L347
                    (metrics.ascent / 2.0, units_per_em / 18.0)
                };

            // Compute line height
            let style = &self.styles[style_index as usize];
            let line_height = match style.line_height {
                LineHeight::Absolute(value) => value,
                LineHeight::FontSizeRelative(value) => value * font_size,
                LineHeight::MetricsRelative(value) => {
                    (metrics.ascent - metrics.descent + metrics.leading) * value
                }
            };

            RunMetrics {
                ascent: metrics.ascent,
                descent: -metrics.descent,
                leading: metrics.leading,
                underline_offset,
                underline_size,
                strikethrough_offset,
                strikethrough_size,
                line_height,
            }
        };

        let cluster_range = self.clusters.len()..self.clusters.len();

        let mut run = RunData {
            font_index,
            font_size,
            synthesis,
            coords_range: coords_start..coords_end,
            text_range,
            bidi_level,
            cluster_range,
            glyph_start: self.glyphs.len(),
            metrics,
            word_spacing,
            letter_spacing,
            advance: 0.,
        };

        // `HarfRust` returns glyphs in visual order, so we need to process them as such while
        // maintaining logical ordering of clusters.

        let glyph_infos = glyph_buffer.glyph_infos();
        if glyph_infos.is_empty() {
            return Vec::new();
        }
        let glyph_positions = glyph_buffer.glyph_positions();
        let scale_factor = font_size / units_per_em;
        let cluster_range_start = self.clusters.len();
        let is_rtl = bidi_level & 1 == 1;
        let glyph_start = self.glyphs.len();
        if !is_rtl {
            let advance = process_clusters(
                Direction::Ltr,
                &mut self.clusters,
                &mut self.glyphs,
                scale_factor,
                glyph_infos,
                glyph_positions,
                char_infos,
                source_text.char_indices(),
            );
            run.advance = advance;
        } else {
            let advance = process_clusters(
                Direction::Rtl,
                &mut self.clusters,
                &mut self.glyphs,
                scale_factor,
                glyph_infos,
                glyph_positions,
                char_infos,
                source_text.char_indices().rev(),
            );
            run.advance = advance;
            // Reverse clusters into logical order for RTL
            let clusters_len = self.clusters.len();
            self.clusters[cluster_range_start..clusters_len].reverse();
        };

        run.cluster_range = cluster_range_start..self.clusters.len();
        assert_eq!(run.cluster_range.len(), char_infos.len());

        // Detect holes (clusters with glyph_id == 0) now that clusters are in logical order.
        // This works correctly for both LTR and RTL since clusters are already normalized.
        let holes = detect_holes(
            &self.clusters[run.cluster_range.clone()],
            &self.glyphs[glyph_start..],
        );

        if !run.cluster_range.is_empty() {
            self.runs.push(run);
            self.items.push(LayoutItem {
                kind: LayoutItemKind::TextRun,
                index: self.runs.len() - 1,
                bidi_level,
            });
        }

        holes
    }

    /// Remove the last pushed run along with its clusters and glyphs.
    /// Used when a run has holes and needs to be re-shaped with fallback fonts.
    #[allow(dead_code)]
    pub(crate) fn undo_last_run(&mut self) {
        // Remove the last item (should be a TextRun)
        if let Some(last_item) = self.items.pop() {
            debug_assert_eq!(last_item.kind, LayoutItemKind::TextRun);
        }

        // Remove the last run and its associated data
        if let Some(run) = self.runs.pop() {
            // Remove clusters for this run
            self.clusters.truncate(run.cluster_range.start);
            // Remove glyphs for this run
            self.glyphs.truncate(run.glyph_start);
            // Remove coords for this run
            self.coords.truncate(run.coords_range.start);
        }
    }

    /// Process a glyph buffer into a `ProcessedRun` WITHOUT pushing to layout.
    /// This allows analyzing for holes before deciding how to push.
    ///
    /// The `cached_metrics` are computed by the shaper with LRU caching.
    /// The `clusters` and `glyphs` vecs are provided from allocation pools.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn process_run_to_temp(
        &mut self,
        font: FontData,
        font_size: f32,
        synthesis: fontique::Synthesis,
        glyph_buffer: &harfrust::GlyphBuffer,
        bidi_level: u8,
        style_index: u16,
        word_spacing: f32,
        letter_spacing: f32,
        source_text: &str,
        char_infos: &[(CharInfo, u16)],
        text_range: Range<usize>,
        coords: &[i16],
        cached_metrics: &crate::shape::cache::CachedMetrics,
        mut clusters: Vec<ClusterData>,
        mut glyphs: Vec<Glyph>,
    ) -> ProcessedRun {
        let font_index = self
            .fonts
            .iter()
            .position(|f| *f == font)
            .unwrap_or_else(|| {
                let index = self.fonts.len();
                self.fonts.push(font);
                index
            });

        // Compute line height from style
        let style = &self.styles[style_index as usize];
        let line_height = match style.line_height {
            LineHeight::Absolute(value) => value,
            LineHeight::FontSizeRelative(value) => value * font_size,
            LineHeight::MetricsRelative(value) => {
                (cached_metrics.ascent + cached_metrics.descent + cached_metrics.leading) * value
            }
        };
        let run_metrics = cached_metrics.to_run_metrics(line_height);

        let glyph_infos = glyph_buffer.glyph_infos();
        if glyph_infos.is_empty() {
            return ProcessedRun {
                clusters,
                glyphs,
                advance: 0.0,
                metrics: run_metrics,
                font_index,
                font_size,
                synthesis,
                coords: coords.to_vec(),
                bidi_level,
                style_index,
                word_spacing,
                letter_spacing,
                text_range,
            };
        }

        let glyph_positions = glyph_buffer.glyph_positions();
        let scale_factor = font_size / cached_metrics.units_per_em;
        let is_rtl = bidi_level & 1 == 1;

        let advance = if !is_rtl {
            process_clusters(
                Direction::Ltr,
                &mut clusters,
                &mut glyphs,
                scale_factor,
                glyph_infos,
                glyph_positions,
                char_infos,
                source_text.char_indices(),
            )
        } else {
            let advance = process_clusters(
                Direction::Rtl,
                &mut clusters,
                &mut glyphs,
                scale_factor,
                glyph_infos,
                glyph_positions,
                char_infos,
                source_text.char_indices().rev(),
            );
            // Reverse clusters into logical order for RTL
            clusters.reverse();
            advance
        };

        ProcessedRun {
            clusters,
            glyphs,
            advance,
            metrics: run_metrics,
            font_index,
            font_size,
            synthesis,
            coords: coords.to_vec(),
            bidi_level,
            style_index,
            word_spacing,
            letter_spacing,
            text_range,
        }
    }

    /// Analyzes a glyph buffer and returns segments (shaped vs holes) in text order.
    /// This is used to split shaped data for hole-based font fallback.
    pub(crate) fn analyze_processed_run(run: &ProcessedRun, text_len: usize) -> ShapedRunAnalysis {
        let clusters = &run.clusters;
        let glyphs = &run.glyphs;
        let holes = detect_holes(clusters, glyphs);

        if holes.is_empty() {
            // No holes - everything is shaped
            return ShapedRunAnalysis {
                segments: vec![ShapedSegment::Shaped {
                    cluster_range: 0..clusters.len(),
                    text_range: 0..text_len,
                }],
                has_holes: false,
            };
        }

        // Build segments from holes
        let mut segments = Vec::new();
        let mut last_end_cluster = 0;
        let mut last_end_text = 0;

        for hole in holes {
            // Add shaped segment before this hole (if any)
            if hole.char_range.start > last_end_cluster {
                segments.push(ShapedSegment::Shaped {
                    cluster_range: last_end_cluster..hole.char_range.start,
                    text_range: last_end_text..hole.text_range.start,
                });
            }

            // Add the hole
            segments.push(ShapedSegment::Hole(hole.clone()));

            last_end_cluster = hole.char_range.end;
            last_end_text = hole.text_range.end;
        }

        // Add trailing shaped segment (if any)
        if last_end_cluster < clusters.len() {
            segments.push(ShapedSegment::Shaped {
                cluster_range: last_end_cluster..clusters.len(),
                text_range: last_end_text..text_len,
            });
        }

        ShapedRunAnalysis {
            segments,
            has_holes: true,
        }
    }

    /// Push a segment from a ProcessedRun to the layout.
    /// The segment_cluster_range is the range of clusters within the ProcessedRun to push.
    /// text_range is the absolute text range for this segment.
    pub(crate) fn push_processed_segment(
        &mut self,
        run: &ProcessedRun,
        segment_cluster_range: Range<usize>,
        text_range: Range<usize>,
    ) {
        if segment_cluster_range.is_empty() {
            return;
        }

        let coords_start = self.coords.len();
        self.coords.extend(run.coords.iter().copied());
        let coords_end = self.coords.len();

        let cluster_range_start = self.clusters.len();
        let glyph_start = self.glyphs.len();

        // Copy the relevant clusters and glyphs
        let segment_clusters = &run.clusters[segment_cluster_range.clone()];

        // Calculate glyph range from cluster data
        let mut segment_advance = 0.0f32;
        let mut min_glyph_offset = usize::MAX;
        let mut max_glyph_end = 0usize;

        for cluster in segment_clusters {
            segment_advance += cluster.advance;
            if cluster.glyph_len != 0xFF && cluster.glyph_len > 0 {
                let start = cluster.glyph_offset as usize;
                let end = start + cluster.glyph_len as usize;
                min_glyph_offset = min_glyph_offset.min(start);
                max_glyph_end = max_glyph_end.max(end);
            }
        }

        // Copy glyphs and adjust cluster glyph offsets
        let glyph_base_offset = if min_glyph_offset < usize::MAX {
            // Copy the relevant glyphs
            self.glyphs
                .extend_from_slice(&run.glyphs[min_glyph_offset..max_glyph_end]);
            min_glyph_offset
        } else {
            0
        };

        // Calculate text offset adjustment (first cluster's text_offset is the base)
        let text_base_offset = segment_clusters
            .first()
            .map(|c| c.text_offset as usize)
            .unwrap_or(0);

        // Copy clusters with adjusted offsets
        for cluster in segment_clusters {
            let mut new_cluster = *cluster;

            // Adjust glyph offset if not inlined
            if new_cluster.glyph_len != 0xFF && new_cluster.glyph_len > 0 {
                new_cluster.glyph_offset =
                    (new_cluster.glyph_offset as usize - glyph_base_offset) as u32;
            }

            // Adjust text offset relative to segment start
            new_cluster.text_offset = (new_cluster.text_offset as usize - text_base_offset) as u16;

            self.clusters.push(new_cluster);
        }

        let run_data = RunData {
            font_index: run.font_index,
            font_size: run.font_size,
            synthesis: run.synthesis.clone(),
            coords_range: coords_start..coords_end,
            text_range,
            bidi_level: run.bidi_level,
            cluster_range: cluster_range_start..self.clusters.len(),
            glyph_start,
            metrics: run.metrics.clone(),
            word_spacing: run.word_spacing,
            letter_spacing: run.letter_spacing,
            advance: segment_advance,
        };

        self.runs.push(run_data);
        self.items.push(LayoutItem {
            kind: LayoutItemKind::TextRun,
            index: self.runs.len() - 1,
            bidi_level: run.bidi_level,
        });
    }

    /// Push an entire ProcessedRun to the layout (no splitting).
    pub(crate) fn push_processed_run(&mut self, run: &ProcessedRun) {
        self.push_processed_segment(run, 0..run.clusters.len(), run.text_range.clone());
    }

    pub(crate) fn finish(&mut self) {
        for run in &self.runs {
            let word = run.word_spacing;
            let letter = run.letter_spacing;
            if nearly_zero(word) && nearly_zero(letter) {
                continue;
            }
            let clusters = &mut self.clusters[run.cluster_range.clone()];
            for cluster in clusters {
                let mut spacing = letter;
                if !nearly_zero(word) && cluster.info.whitespace().is_space_or_nbsp() {
                    spacing += word;
                }
                if !nearly_zero(spacing) {
                    cluster.advance += spacing;
                    if cluster.glyph_len != 0xFF {
                        let start = run.glyph_start + cluster.glyph_offset as usize;
                        let end = start + cluster.glyph_len as usize;
                        let glyphs = &mut self.glyphs[start..end];
                        if let Some(last) = glyphs.last_mut() {
                            last.advance += spacing;
                        }
                    }
                }
            }
        }
    }

    // TODO: this method does not handle mixed direction text at all.
    pub(crate) fn calculate_content_widths(&self) -> ContentWidths {
        fn whitespace_advance(cluster: Option<&ClusterData>) -> f32 {
            cluster
                .filter(|cluster| cluster.info.whitespace().is_space_or_nbsp())
                .map_or(0.0, |cluster| cluster.advance)
        }

        let mut min_width = 0.0_f32;
        let mut max_width = 0.0_f32;

        let mut running_min_width = 0.0;
        let mut running_max_width = 0.0;
        let mut text_wrap_mode = TextWrapMode::Wrap;
        let mut prev_cluster: Option<&ClusterData> = None;
        let is_rtl = self.base_level & 1 == 1;
        for item in &self.items {
            match item.kind {
                LayoutItemKind::TextRun => {
                    let run = &self.runs[item.index];
                    let clusters = &self.clusters[run.cluster_range.clone()];
                    if is_rtl {
                        prev_cluster = clusters.first();
                    }
                    for cluster in clusters {
                        let boundary = cluster.info.boundary();
                        let style = &self.styles[cluster.style_index as usize];
                        let prev_text_wrap_mode = text_wrap_mode;
                        text_wrap_mode = style.text_wrap_mode;
                        if boundary == Boundary::Mandatory
                            || (prev_text_wrap_mode == TextWrapMode::Wrap
                                && (boundary == Boundary::Line
                                    || style.overflow_wrap == OverflowWrap::Anywhere))
                        {
                            let trailing_whitespace = whitespace_advance(prev_cluster);
                            min_width = min_width.max(running_min_width - trailing_whitespace);
                            running_min_width = 0.0;
                            if boundary == Boundary::Mandatory {
                                running_max_width = 0.0;
                            }
                        }
                        running_min_width += cluster.advance;
                        running_max_width += cluster.advance;
                        if !is_rtl {
                            prev_cluster = Some(cluster);
                        }
                    }
                    let trailing_whitespace = whitespace_advance(prev_cluster);
                    min_width = min_width.max(running_min_width - trailing_whitespace);
                }
                LayoutItemKind::InlineBox => {
                    let ibox = &self.inline_boxes[item.index];
                    running_max_width += ibox.width;
                    if text_wrap_mode == TextWrapMode::Wrap {
                        let trailing_whitespace = whitespace_advance(prev_cluster);
                        min_width = min_width.max(running_min_width - trailing_whitespace);
                        min_width = min_width.max(ibox.width);
                        running_min_width = 0.0;
                    } else {
                        running_min_width += ibox.width;
                    }
                    prev_cluster = None;
                }
            }
            let trailing_whitespace = whitespace_advance(prev_cluster);
            max_width = max_width.max(running_max_width - trailing_whitespace);
        }

        let trailing_whitespace = whitespace_advance(prev_cluster);
        min_width = min_width.max(running_min_width - trailing_whitespace);

        ContentWidths {
            min: min_width,
            max: max_width,
        }
    }
}

/// Detects holes (clusters with glyph_id == 0) in the processed clusters.
/// This is called after clusters are in logical order, so it works correctly for both LTR and RTL.
///
/// Returns a list of holes, where each hole is a contiguous range of clusters that have
/// glyph_id == 0 (meaning the font couldn't render those characters).
///
/// **Important**: Combining marks are always included with their preceding base character's hole,
/// even if the font claims to support the mark in isolation. This ensures proper rendering
/// of composed characters like Arabic letters with diacritics.
///
/// Note: `glyphs` should be the slice of glyphs for this run only (not the entire layout).
fn detect_holes(clusters: &[ClusterData], glyphs: &[Glyph]) -> Vec<Hole> {
    let mut holes = Vec::new();
    let mut hole_start: Option<usize> = None; // char index where hole starts
    let mut hole_text_start: Option<usize> = None; // text offset where hole starts

    // First pass: determine which clusters are holes, handling ligature components.
    // A ligature component (glyph_len=0) is a hole if its "owner" cluster is a hole.
    let mut is_hole_vec: Vec<bool> = Vec::with_capacity(clusters.len());

    for cluster in clusters.iter() {
        is_hole_vec.push(cluster_is_hole(cluster, glyphs));
    }

    // Handle ligature components: if cluster[i] has glyph_len=0, it's part of a ligature/combining sequence.
    // The "owner" cluster (the one with the actual glyphs) could be before OR after this cluster:
    // - LTR ligatures (e.g., "fi"): owner ('f') comes BEFORE component ('i')
    // - Arabic combining marks: owner (fatha) might come AFTER base (alef)
    // We look in both directions to find the nearest owner.
    for i in 0..clusters.len() {
        if clusters[i].glyph_len == 0 {
            // Look backward first (for LTR ligatures)
            let mut found = false;
            if i > 0 {
                for j in (0..i).rev() {
                    if clusters[j].glyph_len != 0 {
                        is_hole_vec[i] = is_hole_vec[j];
                        found = true;
                        break;
                    }
                }
            }
            // TODO: This isn't required
            // If not found backward, look forward (for combining marks)
            if !found {
                for j in (i + 1)..clusters.len() {
                    if clusters[j].glyph_len != 0 {
                        is_hole_vec[i] = is_hole_vec[j];
                        break;
                    }
                }
            }
        }
    }

    for (i, cluster) in clusters.iter().enumerate() {
        let is_hole = is_hole_vec[i];
        let is_combining = cluster.advance == 0.0;

        // A cluster is considered part of a hole if:
        // 1. It has glyph_id == 0 (font can't render it), OR
        // 2. It's a combining mark and the previous cluster was a hole
        //    (combining marks should stay with their base character)
        let should_be_in_hole = is_hole || (is_combining && hole_start.is_some());

        if should_be_in_hole {
            // Start or continue a hole
            if hole_start.is_none() {
                hole_start = Some(i);
                hole_text_start = Some(cluster.text_offset as usize);
            }
        } else if let Some(start_idx) = hole_start.take() {
            // End the current hole
            let text_start = hole_text_start.take().unwrap();
            let text_end = cluster.text_offset as usize;
            holes.push(Hole {
                char_range: start_idx..i,
                text_range: text_start..text_end,
            });
        }
    }

    // Handle hole at the end
    if let Some(start_idx) = hole_start {
        let text_start = hole_text_start.unwrap();
        // For the last hole, compute end from the last cluster
        if let Some(last_cluster) = clusters.last() {
            let text_end = last_cluster.text_offset as usize + last_cluster.text_len as usize;
            holes.push(Hole {
                char_range: start_idx..clusters.len(),
                text_range: text_start..text_end,
            });
        }
    }

    holes
}

/// Check if a cluster is a "hole" (font couldn't render it, glyph_id == 0).
/// Note: `glyphs` should be the slice of glyphs for this run (already offset by glyph_base).
fn cluster_is_hole(cluster: &ClusterData, glyphs: &[Glyph]) -> bool {
    let is_hole = if cluster.glyph_len == 0xFF {
        // Single glyph inlined - glyph_offset IS the glyph ID
        cluster.glyph_offset == 0
    } else if cluster.glyph_len == 0 {
        // No glyphs (ligature component) - not a hole by itself
        false
    } else {
        // Multiple glyphs - check if ALL are .notdef (id == 0)
        // glyph_offset is already relative to the run's glyph_start
        let start = cluster.glyph_offset as usize;
        let end = start + cluster.glyph_len as usize;
        if end <= glyphs.len() {
            glyphs[start..end].iter().all(|g| g.id == 0)
        } else {
            false
        }
    };

    is_hole
}

/// Processes shaped glyphs from `HarfRust` and converts them into `ClusterData` and `Glyph`.
///
/// # Parameters
///
/// ## Output Parameters (mutated by this function):
/// * `clusters` - Vector where new `ClusterData` entries will be pushed.
/// * `glyphs` - Vector where new `Glyph` entries will be pushed. Note: single-glyph clusters
///   with zero offsets may be inlined directly into `ClusterData`.
///
/// ## Input Parameters:
/// * `direction` - Direction of the text.
/// * `scale_factor` - Scaling factor used to convert font units to the target size.
/// * `glyph_infos` - `HarfRust` glyph information in visual order.
/// * `glyph_positions` - `HarfRust` glyph positioning data in visual order.
/// * `char_infos` - Character information from text analysis, indexed by cluster ID.
/// * `char_indices_iter` - Iterator over (`byte_offset`, `char`) pairs from the source text.
///   Should be in logical order (forward for LTR, reverse for RTL).
///
/// # Returns
/// * `f32` - Total advance of the run.
fn process_clusters<I: Iterator<Item = (usize, char)>>(
    direction: Direction,
    clusters: &mut Vec<ClusterData>,
    glyphs: &mut Vec<Glyph>,
    scale_factor: f32,
    glyph_infos: &[harfrust::GlyphInfo],
    glyph_positions: &[harfrust::GlyphPosition],
    char_infos: &[(CharInfo, u16)],
    char_indices_iter: I,
) -> f32 {
    let mut char_indices_iter = char_indices_iter.peekable();
    let mut cluster_start_char = char_indices_iter.next().unwrap();
    let mut total_glyphs: u32 = 0;
    let mut cluster_glyph_offset: u32 = 0;
    let start_cluster_id = glyph_infos.first().unwrap().cluster;
    let mut cluster_id = start_cluster_id;
    let mut char_info = char_infos[cluster_id as usize];
    let mut run_advance = 0.0;
    let mut cluster_advance = 0.0;
    // If the current cluster might be a single-glyph, zero-offset cluster, we defer
    // pushing the first glyph to `glyphs` because it might be inlined into `ClusterData`.
    let mut pending_inline_glyph: Option<Glyph> = None;

    // The mental model for understanding this function is best grasped by first reading
    // the HarfBuzz docs on [clusters](https://harfbuzz.github.io/working-with-harfbuzz-clusters.html).
    //
    // `num_components` is the number of characters in the current cluster. Since source text's characters
    // were inserted into `HarfRust`'s buffer using their logical indices as the cluster ID, `HarfRust` will
    // assign the first character's cluster ID (in logical order) to the merged cluster because the minimum
    // ID is selected for [merging](https://github.com/harfbuzz/harfrust/blob/a38025fb336230b492366740c86021bb406bcd0d/src/hb/buffer.rs#L920-L924).
    //
    //  So, the number of components in a given cluster is dependent on `direction`.
    //   - In LTR, `num_components` is the difference between the next cluster and the current cluster.
    //   - In RTL, `num_components` is the difference between the last cluster and the current cluster.
    // This is because we must compare the current cluster to its next larger ID (in other words, the next
    // logical index, which is visually downstream in LTR and visually upstream in RTL).
    //
    // For example, consider the LTR text for "afi" where "fi" form a ligature.
    //   Initial cluster values: 0, 1, 2 (logical + visual order)
    //   `HarfRust` assignation: 0, 1, 1
    //   Cluster count:          2
    //   `num_components`:       (1 - 0 =) 1, (3 - 1 =) 2
    //
    // Now consider the RTL text for "حداً".
    //   Initial cluster values:  0, 1, 2, 3 (logical, or in-memory, order)
    //   Reversed cluster values: 3, 2, 1, 0 (visual order - the return order of `HarfRust` for RTL)
    //   `HarfRust` assignation:  3, 2, 0, 0
    //   Cluster count:           3
    //   `num_components`:        (4 - 3 =) 1, (3 - 2 =) 1, (2 - 0 =) 2
    let num_components =
        |next_cluster: u32, current_cluster: u32, last_cluster: u32| match direction {
            Direction::Ltr => next_cluster - current_cluster,
            Direction::Rtl => last_cluster - current_cluster,
        };
    let mut last_cluster_id: u32 = match direction {
        Direction::Ltr => 0,
        Direction::Rtl => char_infos.len() as u32,
    };

    for (glyph_info, glyph_pos) in glyph_infos.iter().zip(glyph_positions.iter()) {
        // Flush previous cluster if we've reached a new cluster
        if cluster_id != glyph_info.cluster {
            run_advance += cluster_advance;
            let num_components = num_components(glyph_info.cluster, cluster_id, last_cluster_id);
            cluster_advance /= num_components as f32;
            let is_newline = to_whitespace(cluster_start_char.1) == Whitespace::Newline;
            let cluster_type = if num_components > 1 {
                debug_assert!(!is_newline);
                ClusterType::LigatureStart
            } else if is_newline {
                ClusterType::Newline
            } else {
                ClusterType::Regular
            };

            let inline_glyph_id = if matches!(cluster_type, ClusterType::Regular) {
                pending_inline_glyph.take().map(|g| g.id)
            } else {
                // This isn't a regular cluster, so we don't inline the glyph and push
                // it to `glyphs`.
                if let Some(pending) = pending_inline_glyph.take() {
                    glyphs.push(pending);
                    total_glyphs += 1;
                }
                None
            };

            push_cluster(
                clusters,
                char_info,
                cluster_start_char,
                cluster_glyph_offset,
                cluster_advance,
                total_glyphs,
                cluster_type,
                inline_glyph_id,
            );
            cluster_glyph_offset = total_glyphs;

            if num_components > 1 {
                // Skip characters until we reach the current cluster
                for i in 1..num_components {
                    cluster_start_char = char_indices_iter.next().unwrap();
                    if to_whitespace(cluster_start_char.1) == Whitespace::Space {
                        break;
                    }
                    let char_info_ = match direction {
                        Direction::Ltr => char_infos[(cluster_id + i) as usize],
                        Direction::Rtl => char_infos[(cluster_id + num_components - i) as usize],
                    };
                    push_cluster(
                        clusters,
                        char_info_,
                        cluster_start_char,
                        cluster_glyph_offset,
                        cluster_advance,
                        total_glyphs,
                        ClusterType::LigatureComponent,
                        None,
                    );
                }
            }
            // Hole tracking is currently disabled.
            // TODO: Re-enable once RTL handling is properly implemented.

            cluster_start_char = char_indices_iter.next().unwrap();

            cluster_advance = 0.0;
            last_cluster_id = cluster_id;
            cluster_id = glyph_info.cluster;
            char_info = char_infos[cluster_id as usize];
            pending_inline_glyph = None;
        }

        let glyph = Glyph {
            id: glyph_info.glyph_id,
            style_index: char_info.1,
            x: (glyph_pos.x_offset as f32) * scale_factor,
            y: (glyph_pos.y_offset as f32) * scale_factor,
            advance: (glyph_pos.x_advance as f32) * scale_factor,
        };
        cluster_advance += glyph.advance;
        // Push any pending glyph. If it was a zero-offset, single glyph cluster, it would
        // have been pushed in the first `if` block.
        if let Some(pending) = pending_inline_glyph.take() {
            glyphs.push(pending);
            total_glyphs += 1;
        }
        if total_glyphs == cluster_glyph_offset && glyph.x == 0.0 && glyph.y == 0.0 {
            // Defer this potential zero-offset, single glyph cluster
            pending_inline_glyph = Some(glyph);
        } else {
            glyphs.push(glyph);
            total_glyphs += 1;
        }
    }

    // Push the last cluster
    {
        // See comment above `num_components` for why we use `char_infos.len()` for LTR and 0 for RTL.
        let next_cluster_id = match direction {
            Direction::Ltr => char_infos.len() as u32,
            Direction::Rtl => 0,
        };
        let num_components = num_components(next_cluster_id, cluster_id, last_cluster_id);
        if num_components > 1 {
            // This is a ligature - create ligature start + ligature components

            if let Some(pending) = pending_inline_glyph.take() {
                glyphs.push(pending);
                total_glyphs += 1;
            }
            let ligature_advance = cluster_advance / num_components as f32;
            push_cluster(
                clusters,
                char_info,
                cluster_start_char,
                cluster_glyph_offset,
                ligature_advance,
                total_glyphs,
                ClusterType::LigatureStart,
                None,
            );

            cluster_glyph_offset = total_glyphs;

            // Create ligature component clusters for the remaining characters
            let mut i = 1;
            for char in char_indices_iter {
                if to_whitespace(char.1) == Whitespace::Space {
                    break;
                }
                let component_char_info = match direction {
                    Direction::Ltr => char_infos[(cluster_id + i) as usize],
                    Direction::Rtl => char_infos[(cluster_id + num_components - i) as usize],
                };
                push_cluster(
                    clusters,
                    component_char_info,
                    char,
                    cluster_glyph_offset,
                    ligature_advance,
                    total_glyphs,
                    ClusterType::LigatureComponent,
                    None,
                );
                i += 1;
            }
        } else {
            let is_newline = to_whitespace(cluster_start_char.1) == Whitespace::Newline;
            let cluster_type = if is_newline {
                ClusterType::Newline
            } else {
                ClusterType::Regular
            };
            let mut inline_glyph_id = None;
            match cluster_type {
                ClusterType::Regular => {
                    if total_glyphs == cluster_glyph_offset {
                        if let Some(pending) = pending_inline_glyph.take() {
                            inline_glyph_id = Some(pending.id);
                        }
                    }
                }
                _ => {
                    if let Some(pending) = pending_inline_glyph.take() {
                        glyphs.push(pending);
                        total_glyphs += 1;
                    }
                }
            }
            push_cluster(
                clusters,
                char_info,
                cluster_start_char,
                cluster_glyph_offset,
                cluster_advance,
                total_glyphs,
                cluster_type,
                inline_glyph_id,
            );
        }
    }

    run_advance
}

#[derive(PartialEq)]
enum Direction {
    Ltr,
    Rtl,
}

enum ClusterType {
    LigatureStart,
    LigatureComponent,
    Regular,
    Newline,
}

impl From<&ClusterType> for u16 {
    fn from(cluster_type: &ClusterType) -> Self {
        match cluster_type {
            ClusterType::LigatureStart => ClusterData::LIGATURE_START,
            ClusterType::LigatureComponent => ClusterData::LIGATURE_COMPONENT,
            ClusterType::Regular | ClusterType::Newline => 0, // No special flags
        }
    }
}

fn push_cluster(
    clusters: &mut Vec<ClusterData>,
    char_info: (CharInfo, u16),
    cluster_start_char: (usize, char),
    glyph_offset: u32,
    advance: f32,
    total_glyphs: u32,
    cluster_type: ClusterType,
    inline_glyph_id: Option<u32>,
) {
    let glyph_len = (total_glyphs - glyph_offset) as u8;

    let (final_glyph_len, final_glyph_offset, final_advance) = match cluster_type {
        ClusterType::LigatureComponent => {
            // Ligature components have no glyphs, only advance.
            debug_assert_eq!(glyph_len, 0);
            (0_u8, 0_u32, advance)
        }
        ClusterType::Newline => {
            // Newline clusters are stripped of their glyph contribution.
            debug_assert_eq!(glyph_len, 1);
            (0_u8, 0_u32, 0.0)
        }
        _ if inline_glyph_id.is_some() => {
            // Inline glyphs are stored inline within `ClusterData`
            debug_assert_eq!(glyph_len, 0);
            (0xFF_u8, inline_glyph_id.unwrap(), advance)
        }
        ClusterType::Regular | ClusterType::LigatureStart => {
            // Regular and ligature start clusters maintain their glyphs and advance.
            debug_assert_ne!(glyph_len, 0);
            (glyph_len, glyph_offset, advance)
        }
    };

    clusters.push(ClusterData {
        info: ClusterInfo::new(char_info.0.boundary, cluster_start_char.1),
        flags: (&cluster_type).into(),
        style_index: char_info.1,
        glyph_len: final_glyph_len,
        text_len: cluster_start_char.1.len_utf8() as u8,
        glyph_offset: final_glyph_offset,
        text_offset: cluster_start_char.0 as u16,
        advance: final_advance,
    });
}
