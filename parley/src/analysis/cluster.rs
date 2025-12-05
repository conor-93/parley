// Copyright 2025 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

/// The maximum number of characters in a single cluster.
const MAX_CLUSTER_SIZE: usize = 32;

#[derive(Debug, Default)]
pub(crate) struct CharCluster {
    pub style_index: u16,
    pub is_emoji: bool,
    pub start: u32,
    pub end: u32,
}

impl CharCluster {
    pub(crate) fn range(&self) -> SourceRange {
        SourceRange {
            start: self.start,
            end: self.end,
        }
    }
}

/// Source range of a cluster in code units.
#[derive(Copy, Clone)]
pub(crate) struct SourceRange {
    pub start: u32,
    pub end: u32,
}

/// Whitespace content of a cluster.
#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum Whitespace {
    /// Not a space.
    None = 0,
    /// Standard space.
    Space = 1,
    /// Non-breaking space (U+00A0).
    NoBreakSpace = 2,
    /// Horizontal tab.
    Tab = 3,
    /// Newline (CR, LF, or CRLF).
    Newline = 4,
}

impl Whitespace {
    /// Returns true for space or no break space.
    pub(crate) fn is_space_or_nbsp(self) -> bool {
        matches!(self, Self::Space | Self::NoBreakSpace)
    }
}

impl CharCluster {
    pub(crate) fn clear(&mut self) {
        self.style_index = u16::MAX;
        self.is_emoji = false;
        self.start = 0;
        self.end = 0;
    }

    /// Returns the primary style index for the cluster.
    pub(crate) fn style_index(&self) -> u16 {
        self.style_index
    }
}
