use crate::config::PackerConfig;
use crate::config::{AlgorithmFamily, AutoMode, SortOrder};
use crate::error::{Result, TexPackerError};
use crate::model::{Atlas, Frame, Meta, Page, Rect};
use crate::packer::{
    Packer, guillotine::GuillotinePacker, maxrects::MaxRectsPacker, skyline::SkylinePacker,
};
use image::{DynamicImage, GenericImageView, RgbaImage};
use std::collections::{HashMap, HashSet};
use std::ops::Index;
use std::time::Instant;
use tracing::instrument;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// In-memory image to pack (key + decoded image).
pub struct InputImage {
    pub key: String,
    pub image: DynamicImage,
}

/// Output RGBA page and its logical page record.
pub struct OutputPage {
    pub page: Page,
    pub rgba: RgbaImage,
}

/// Output of a packing run: atlas metadata and RGBA pages.
pub struct PackOutput {
    pub atlas: Atlas,
    pub pages: Vec<OutputPage>,
}

impl PackOutput {
    /// Computes packing statistics for this output.
    /// This is a convenience method that delegates to `atlas.stats()`.
    pub fn stats(&self) -> crate::model::PackStats {
        self.atlas.stats()
    }
}

#[instrument(skip_all)]
/// Packs `inputs` into atlas pages using configuration `cfg` and returns metadata and RGBA pages.
///
/// Notes:
/// - Sorting is stable for deterministic results.
/// - When `family` is `Auto`, a small portfolio is tried and the best result is chosen (pages first, then total area).
/// - `time_budget_ms` can limit Auto evaluation time; `parallel` may evaluate in parallel when enabled.
pub fn pack_images(inputs: Vec<InputImage>, cfg: PackerConfig) -> Result<PackOutput> {
    // Validate configuration first
    cfg.validate()?;

    if inputs.is_empty() {
        return Err(TexPackerError::Empty);
    }

    // Preprocess once
    let prepared = prepare_inputs(&inputs, &cfg);

    // Auto portfolio
    if matches!(cfg.family, AlgorithmFamily::Auto) {
        return pack_auto(&prepared, cfg);
    }

    pack_prepared(&prepared, &cfg)
}

pub fn compute_trim_rect(rgba: &RgbaImage, threshold: u8) -> (Option<Rect>, Rect) {
    let (w, h) = rgba.dimensions();
    let mut x1 = 0;
    let mut y1 = 0;
    let mut x2 = w.saturating_sub(1);
    let mut y2 = h.saturating_sub(1);
    // left
    while x1 < w {
        let mut all_transparent = true;
        for y in 0..h {
            if rgba.get_pixel(x1, y)[3] > threshold {
                all_transparent = false;
                break;
            }
        }
        if all_transparent {
            x1 += 1;
        } else {
            break;
        }
    }
    if x1 >= w {
        return (None, Rect::new(0, 0, w, h));
    }
    // right
    while x2 > x1 {
        let mut all_transparent = true;
        for y in 0..h {
            if rgba.get_pixel(x2, y)[3] > threshold {
                all_transparent = false;
                break;
            }
        }
        if all_transparent {
            x2 -= 1;
        } else {
            break;
        }
    }
    // top
    while y1 < h {
        let mut all_transparent = true;
        for x in x1..=x2 {
            if rgba.get_pixel(x, y1)[3] > threshold {
                all_transparent = false;
                break;
            }
        }
        if all_transparent {
            y1 += 1;
        } else {
            break;
        }
    }
    // bottom
    while y2 > y1 {
        let mut all_transparent = true;
        for x in x1..=x2 {
            if rgba.get_pixel(x, y2)[3] > threshold {
                all_transparent = false;
                break;
            }
        }
        if all_transparent {
            y2 -= 1;
        } else {
            break;
        }
    }
    let tw = x2 - x1 + 1;
    let th = y2 - y1 + 1;
    (Some(Rect::new(0, 0, tw, th)), Rect::new(x1, y1, tw, th))
}

fn next_pow2(mut v: u32) -> u32 {
    if v <= 1 {
        return 1;
    }
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v + 1
}

#[allow(clippy::too_many_arguments)]
// moved to compositing::blit_rgba for reuse in runtime

// ---------- helpers for multi-run (auto) ----------

struct Prep {
    key: String,
    rgba: RgbaImage, // Original full image (needed for blitting)
    content: Vec<u8>, // Cached raw pixels of the sub-rect (for fast comparison/hashing)
    hash: u64,        // Pre-computed hash
    rect: Rect,
    trimmed: bool,
    source: Rect,
    orig_size: (u32, u32),
}

fn prepare_inputs(inputs: &[InputImage], cfg: &PackerConfig) -> Vec<Prep> {
    let mut out = Vec::with_capacity(inputs.len());
    for inp in inputs.iter() {
        let rgba = inp.image.to_rgba8();
        let (iw, ih) = rgba.dimensions();
        let mut push_entry = true;
        let (rect, trimmed, source) = if cfg.trim {
            let (trim_rect_opt, src_rect) = compute_trim_rect(&rgba, cfg.trim_threshold);
            match trim_rect_opt {
                Some(r) => (Rect::new(0, 0, r.w, r.h), true, src_rect),
                None => match cfg.transparent_policy {
                    crate::config::TransparentPolicy::Keep => {
                        (Rect::new(0, 0, iw, ih), false, Rect::new(0, 0, iw, ih))
                    }
                    crate::config::TransparentPolicy::OneByOne => {
                        (Rect::new(0, 0, 1, 1), true, Rect::new(0, 0, 1, 1))
                    }
                    crate::config::TransparentPolicy::Skip => {
                        push_entry = false;
                        (Rect::new(0, 0, 0, 0), false, Rect::new(0, 0, 0, 0))
                    }
                },
            }
        } else {
            (Rect::new(0, 0, iw, ih), false, Rect::new(0, 0, iw, ih))
        };
        if !push_entry {
            continue;
        }
        // Extract the actual pixel data for the sub-rect area once.
        // This is the "content" we care about for deduplication.
        let content = rgba.view(source.x, source.y, source.w, source.h)
            .to_image()
            .into_raw(); // Consumes the sub-image into a flat Vec<u8>

        // Calculate hash using the ultra-fast XXH3 algorithm
        let hash = xxhash_rust::const_xxh3::xxh3_64(&content);
        out.push(Prep {
            key: inp.key.clone(),
            rgba,
            content,
            hash,
            rect,
            trimmed,
            source,
            orig_size: (iw, ih),
        });
    }
    // stable sort per config
    match cfg.sort_order {
        SortOrder::None => {}
        SortOrder::NameAsc => {
            out.sort_by(|a, b| a.key.cmp(&b.key));
        }
        SortOrder::AreaDesc => {
            out.sort_by(|a, b| {
                (b.rect.w * b.rect.h)
                    .cmp(&(a.rect.w * a.rect.h))
                    .then_with(|| a.key.cmp(&b.key))
            });
        }
        SortOrder::MaxSideDesc => {
            out.sort_by(|a, b| {
                b.rect
                    .w
                    .max(b.rect.h)
                    .cmp(&a.rect.w.max(a.rect.h))
                    .then_with(|| a.key.cmp(&b.key))
            });
        }
        SortOrder::HeightDesc => {
            out.sort_by(|a, b| b.rect.h.cmp(&a.rect.h).then_with(|| a.key.cmp(&b.key)));
        }
        SortOrder::WidthDesc => {
            out.sort_by(|a, b| b.rect.w.cmp(&a.rect.w).then_with(|| a.key.cmp(&b.key)));
        }
    }
    out
}


fn pack_prepared(prepared: &[Prep], cfg: &PackerConfig) -> Result<PackOutput> {
    let mut pages: Vec<OutputPage> = Vec::new();
    let mut atlas_pages: Vec<Page> = Vec::new();

    // Map for quick lookup during compositing
    let prep_map: HashMap<String, &Prep> = prepared.iter().map(|p| (p.key.clone(), p)).collect();
    // Stores indices of entry actually placed in the atlas to prevent duplicate rendering (blitting).
    let mut in_packs: HashSet<usize> = HashSet::with_capacity(prepared.len());
    // Remaining indices to place (in sorted order)
    let mut remaining: Vec<usize> = (0..prepared.len()).collect();
    let mut page_id = 0usize;

    while !remaining.is_empty() {
        let mut packer: Box<dyn Packer<String>> = match cfg.family {
            AlgorithmFamily::Skyline => Box::new(SkylinePacker::new(cfg.clone())),
            AlgorithmFamily::MaxRects => {
                Box::new(MaxRectsPacker::new(cfg.clone(), cfg.mr_heuristic.clone()))
            }
            AlgorithmFamily::Guillotine => Box::new(GuillotinePacker::new(
                cfg.clone(),
                cfg.g_choice.clone(),
                cfg.g_split.clone(),
            )),
            AlgorithmFamily::Auto => unreachable!(),
        };
        let mut frames: Vec<Frame> = Vec::new();

        loop {
            let mut placed_any = false;
            let mut remove_set: HashSet<usize> = HashSet::new();
            for &idx in &remaining {
                let p = &prepared[idx];
                let img_data= p.rgba.view(p.source.x,p.source.y,p.source.w,p.source.h);
                if let Some(in_pack_same) = in_packs.iter().find(|x| {
                    let _p: &Prep = &prepared[**x];
                    p.hash == _p.hash && p.content == _p.content
                }){
                    let mut f  = frames[*in_pack_same].clone();
                    f.trimmed = p.trimmed;
                    f.source = p.source;
                    f.source_size = p.orig_size;
                    frames.push(f);
                    remove_set.insert(idx);
                    placed_any = true;
                    continue;
                }

                if !packer.can_pack(&p.rect) {
                    continue;
                }
                if let Some(mut f) = packer.pack(p.key.clone(), &p.rect) {
                    f.trimmed = p.trimmed;
                    f.source = p.source;
                    f.source_size = p.orig_size;
                    frames.push(f);
                    remove_set.insert(idx);
                    in_packs.insert(idx);
                    placed_any = true;
                }
            }
            if !placed_any {
                break;
            }
            // Retain only indices not placed
            if !remove_set.is_empty() {
                remaining.retain(|i| !remove_set.contains(i));
            }
        }

        if frames.is_empty() {
            // No textures could be placed on this page - likely first texture is too large
            let placed = prepared.len() - remaining.len();
            return Err(TexPackerError::OutOfSpaceGeneric {
                placed,
                total: prepared.len(),
            });
        }

        // Compute final page size via helper to keep logic consistent across APIs
        let (page_w, page_h) = compute_page_size(&frames, cfg);

        let mut canvas = RgbaImage::new(page_w, page_h);
        for (i,f) in frames.iter().enumerate() {
            // Only place frames that were actually placed
            if !in_packs.contains(&i){
                continue;
            }
            if let Some(prep) = prep_map.get(&f.key) {
                crate::compositing::blit_rgba(
                    &prep.rgba,
                    &mut canvas,
                    f.frame.x,
                    f.frame.y,
                    prep.source.x,
                    prep.source.y,
                    prep.source.w,
                    prep.source.h,
                    f.rotated,
                    cfg.texture_extrusion,
                    cfg.texture_outlines,
                );
            }
        }
        let page = Page {
            id: page_id,
            width: page_w,
            height: page_h,
            frames: frames.clone(),
        };
        pages.push(OutputPage {
            page: page.clone(),
            rgba: canvas,
        });
        atlas_pages.push(page);
        page_id += 1;
    }

    let meta = Meta {
        schema_version: "1".into(),
        app: "tex-packer".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        format: "RGBA8888".into(),
        scale: 1.0,
        power_of_two: cfg.power_of_two,
        square: cfg.square,
        max_dim: (cfg.max_width, cfg.max_height),
        padding: (cfg.border_padding, cfg.texture_padding),
        extrude: cfg.texture_extrusion,
        allow_rotation: cfg.allow_rotation,
        trim_mode: if cfg.trim { "trim" } else { "none" }.into(),
        background_color: None,
    };
    let atlas = Atlas {
        pages: atlas_pages,
        meta,
    };
    Ok(PackOutput { atlas, pages })
}

fn pack_auto(prepared: &[Prep], base: PackerConfig) -> Result<PackOutput> {
    let mut candidates: Vec<PackerConfig> = Vec::new();
    let n_inputs = prepared.len();
    let budget_ms = base.time_budget_ms.unwrap_or(0);
    let thr_time = base.auto_mr_ref_time_ms_threshold.unwrap_or(200);
    let thr_inputs = base.auto_mr_ref_input_threshold.unwrap_or(800);
    let enable_mr_ref = matches!(base.auto_mode, AutoMode::Quality)
        && (budget_ms >= thr_time || n_inputs >= thr_inputs);
    match base.auto_mode {
        AutoMode::Fast => {
            let mut s_bl = base.clone();
            s_bl.family = AlgorithmFamily::Skyline;
            s_bl.skyline_heuristic = crate::config::SkylineHeuristic::BottomLeft;
            candidates.push(s_bl);
            let mut mr_baf = base.clone();
            mr_baf.family = AlgorithmFamily::MaxRects;
            mr_baf.mr_heuristic = crate::config::MaxRectsHeuristic::BestAreaFit;
            mr_baf.mr_reference = false;
            candidates.push(mr_baf);
        }
        AutoMode::Quality => {
            let mut s_mw = base.clone();
            s_mw.family = AlgorithmFamily::Skyline;
            s_mw.skyline_heuristic = crate::config::SkylineHeuristic::MinWaste;
            candidates.push(s_mw);
            let mut mr_baf = base.clone();
            mr_baf.family = AlgorithmFamily::MaxRects;
            mr_baf.mr_heuristic = crate::config::MaxRectsHeuristic::BestAreaFit;
            mr_baf.mr_reference = enable_mr_ref;
            candidates.push(mr_baf);
            let mut mr_bl = base.clone();
            mr_bl.family = AlgorithmFamily::MaxRects;
            mr_bl.mr_heuristic = crate::config::MaxRectsHeuristic::BottomLeft;
            mr_bl.mr_reference = enable_mr_ref;
            candidates.push(mr_bl);
            let mut mr_cp = base.clone();
            mr_cp.family = AlgorithmFamily::MaxRects;
            mr_cp.mr_heuristic = crate::config::MaxRectsHeuristic::ContactPoint;
            mr_cp.mr_reference = enable_mr_ref;
            candidates.push(mr_cp);
            let mut g = base.clone();
            g.family = AlgorithmFamily::Guillotine;
            g.g_choice = crate::config::GuillotineChoice::BestAreaFit;
            g.g_split = crate::config::GuillotineSplit::SplitShorterLeftoverAxis;
            candidates.push(g);
        }
    }
    let start = Instant::now();

    // Parallel path (optional)
    #[cfg(feature = "parallel")]
    {
        if base.parallel {
            let results: Vec<(PackOutput, u64, u32)> = candidates
                .par_iter()
                .filter_map(|cand| pack_prepared(prepared, cand).ok())
                .map(|out| {
                    let pages = out.atlas.pages.len() as u32;
                    let total_area: u64 = out
                        .atlas
                        .pages
                        .iter()
                        .map(|p| (p.width as u64) * (p.height as u64))
                        .sum();
                    (out, total_area, pages)
                })
                .collect();
            let best = results.into_iter().min_by(|a, b| match a.2.cmp(&b.2) {
                // pages asc
                std::cmp::Ordering::Equal => a.1.cmp(&b.1),
                other => other,
            });
            return best.map(|x| x.0).ok_or(TexPackerError::OutOfSpaceGeneric {
                placed: 0,
                total: prepared.len(),
            });
        }
    }

    // Sequential path with optional time budget
    let mut best: Option<(PackOutput, u64, u32)> = None; // (output, total_area, pages)
    for cand in candidates.into_iter() {
        if budget_ms > 0 && start.elapsed().as_millis() as u64 > budget_ms {
            break;
        }
        if let Ok(out) = pack_prepared(prepared, &cand) {
            let pages = out.atlas.pages.len() as u32;
            let total_area: u64 = out
                .atlas
                .pages
                .iter()
                .map(|p| (p.width as u64) * (p.height as u64))
                .sum();
            match &mut best {
                None => best = Some((out, total_area, pages)),
                Some((bo, barea, bpages)) => {
                    if pages < *bpages || (pages == *bpages && total_area < *barea) {
                        *bo = out;
                        *barea = total_area;
                        *bpages = pages;
                    }
                }
            }
        }
    }
    best.map(|x| x.0).ok_or(TexPackerError::OutOfSpaceGeneric {
        placed: 0,
        total: prepared.len(),
    })
}

// ---------------- Layout-only API ----------------

/// Packs sizes into pages without compositing pixel data.
/// Inputs are (key, width, height). Returns an Atlas with pages and frames; no RGBA pages.
pub fn pack_layout<K: Into<String>>(
    inputs: Vec<(K, u32, u32)>,
    cfg: PackerConfig,
) -> Result<Atlas<String>> {
    // Validate configuration first
    cfg.validate()?;

    if inputs.is_empty() {
        return Err(TexPackerError::Empty);
    }
    // Build lightweight preps
    struct PrepL {
        key: String,
        rect: Rect,
        trimmed: bool,
        source: Rect,
        orig_size: (u32, u32),
    }
    let mut prepared: Vec<PrepL> = inputs
        .into_iter()
        .map(|(k, w, h)| {
            let key = k.into();
            let rect = Rect::new(0, 0, w, h);
            let source = Rect::new(0, 0, w, h);
            PrepL {
                key,
                rect,
                trimmed: false,
                source,
                orig_size: (w, h),
            }
        })
        .collect();
    // Sort like pack_images
    match cfg.sort_order {
        SortOrder::None => {}
        SortOrder::NameAsc => prepared.sort_by(|a, b| a.key.cmp(&b.key)),
        SortOrder::AreaDesc => prepared.sort_by(|a, b| {
            (b.rect.w * b.rect.h)
                .cmp(&(a.rect.w * a.rect.h))
                .then_with(|| a.key.cmp(&b.key))
        }),
        SortOrder::MaxSideDesc => prepared.sort_by(|a, b| {
            b.rect
                .w
                .max(b.rect.h)
                .cmp(&a.rect.w.max(a.rect.h))
                .then_with(|| a.key.cmp(&b.key))
        }),
        SortOrder::HeightDesc => {
            prepared.sort_by(|a, b| b.rect.h.cmp(&a.rect.h).then_with(|| a.key.cmp(&b.key)))
        }
        SortOrder::WidthDesc => {
            prepared.sort_by(|a, b| b.rect.w.cmp(&a.rect.w).then_with(|| a.key.cmp(&b.key)))
        }
    }

    let mut remaining: Vec<usize> = (0..prepared.len()).collect();
    let mut atlas_pages: Vec<Page> = Vec::new();
    let mut page_id = 0usize;
    while !remaining.is_empty() {
        let mut packer: Box<dyn Packer<String>> = match cfg.family {
            AlgorithmFamily::Skyline => Box::new(SkylinePacker::new(cfg.clone())),
            AlgorithmFamily::MaxRects => {
                Box::new(MaxRectsPacker::new(cfg.clone(), cfg.mr_heuristic.clone()))
            }
            AlgorithmFamily::Guillotine => Box::new(GuillotinePacker::new(
                cfg.clone(),
                cfg.g_choice.clone(),
                cfg.g_split.clone(),
            )),
            AlgorithmFamily::Auto => unreachable!(),
        };
        let mut frames: Vec<Frame> = Vec::new();
        loop {
            let mut placed_any = false;
            let mut remove_set: HashSet<usize> = HashSet::new();
            for &idx in &remaining {
                let p = &prepared[idx];
                if !packer.can_pack(&p.rect) {
                    continue;
                }
                if let Some(mut f) = packer.pack(p.key.clone(), &p.rect) {
                    f.trimmed = p.trimmed;
                    f.source = p.source;
                    f.source_size = p.orig_size;
                    frames.push(f);
                    remove_set.insert(idx);
                    placed_any = true;
                }
            }
            if !placed_any {
                break;
            }
            if !remove_set.is_empty() {
                remaining.retain(|i| !remove_set.contains(i));
            }
        }
        if frames.is_empty() {
            let placed = prepared.len() - remaining.len();
            return Err(TexPackerError::OutOfSpaceGeneric {
                placed,
                total: prepared.len(),
            });
        }

        // Compute page size same as pack_prepared
        let (page_w, page_h) = compute_page_size(&frames, &cfg);

        let page = Page {
            id: page_id,
            width: page_w,
            height: page_h,
            frames: frames.clone(),
        };
        atlas_pages.push(page);
        page_id += 1;
    }

    let meta = Meta {
        schema_version: "1".into(),
        app: "tex-packer".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        format: "RGBA8888".into(),
        scale: 1.0,
        power_of_two: cfg.power_of_two,
        square: cfg.square,
        max_dim: (cfg.max_width, cfg.max_height),
        padding: (cfg.border_padding, cfg.texture_padding),
        extrude: cfg.texture_extrusion,
        allow_rotation: cfg.allow_rotation,
        trim_mode: if cfg.trim { "trim" } else { "none" }.into(),
        background_color: None,
    };
    Ok(Atlas {
        pages: atlas_pages,
        meta,
    })
}

/// Layout-only item with optional source/source_size to propagate trimming metadata.
#[derive(Debug, Clone)]
pub struct LayoutItem<K = String> {
    pub key: K,
    pub w: u32,
    pub h: u32,
    pub source: Option<Rect>,
    pub source_size: Option<(u32, u32)>,
    pub trimmed: bool,
}

/// Packs layout-only items (with optional source/source_size metadata) into pages.
pub fn pack_layout_items<K: Into<String>>(
    items: Vec<LayoutItem<K>>,
    cfg: PackerConfig,
) -> Result<Atlas<String>> {
    // Validate configuration first
    cfg.validate()?;

    if items.is_empty() {
        return Err(TexPackerError::Empty);
    }
    struct PrepL {
        key: String,
        rect: Rect,
        trimmed: bool,
        source: Rect,
        orig_size: (u32, u32),
    }
    let mut prepared: Vec<PrepL> = items
        .into_iter()
        .map(|it| {
            let key = it.key.into();
            let rect = Rect::new(0, 0, it.w, it.h);
            let source = it.source.unwrap_or(Rect::new(0, 0, it.w, it.h));
            let orig = it.source_size.unwrap_or((it.w, it.h));
            PrepL {
                key,
                rect,
                trimmed: it.trimmed,
                source,
                orig_size: orig,
            }
        })
        .collect();
    match cfg.sort_order {
        SortOrder::None => {}
        SortOrder::NameAsc => prepared.sort_by(|a, b| a.key.cmp(&b.key)),
        SortOrder::AreaDesc => prepared.sort_by(|a, b| {
            (b.rect.w * b.rect.h)
                .cmp(&(a.rect.w * a.rect.h))
                .then_with(|| a.key.cmp(&b.key))
        }),
        SortOrder::MaxSideDesc => prepared.sort_by(|a, b| {
            b.rect
                .w
                .max(b.rect.h)
                .cmp(&a.rect.w.max(a.rect.h))
                .then_with(|| a.key.cmp(&b.key))
        }),
        SortOrder::HeightDesc => {
            prepared.sort_by(|a, b| b.rect.h.cmp(&a.rect.h).then_with(|| a.key.cmp(&b.key)))
        }
        SortOrder::WidthDesc => {
            prepared.sort_by(|a, b| b.rect.w.cmp(&a.rect.w).then_with(|| a.key.cmp(&b.key)))
        }
    }

    let mut remaining: Vec<usize> = (0..prepared.len()).collect();
    let mut atlas_pages: Vec<Page> = Vec::new();
    let mut page_id = 0usize;
    while !remaining.is_empty() {
        let mut packer: Box<dyn Packer<String>> = match cfg.family {
            AlgorithmFamily::Skyline => Box::new(SkylinePacker::new(cfg.clone())),
            AlgorithmFamily::MaxRects => {
                Box::new(MaxRectsPacker::new(cfg.clone(), cfg.mr_heuristic.clone()))
            }
            AlgorithmFamily::Guillotine => Box::new(GuillotinePacker::new(
                cfg.clone(),
                cfg.g_choice.clone(),
                cfg.g_split.clone(),
            )),
            AlgorithmFamily::Auto => unreachable!(),
        };
        let mut frames: Vec<Frame> = Vec::new();
        loop {
            let mut placed_any = false;
            let mut remove_set: HashSet<usize> = HashSet::new();
            for &idx in &remaining {
                let p = &prepared[idx];
                if !packer.can_pack(&p.rect) {
                    continue;
                }
                if let Some(mut f) = packer.pack(p.key.clone(), &p.rect) {
                    f.trimmed = p.trimmed;
                    f.source = p.source;
                    f.source_size = p.orig_size;
                    frames.push(f);
                    remove_set.insert(idx);
                    placed_any = true;
                }
            }
            if !placed_any {
                break;
            }
            if !remove_set.is_empty() {
                remaining.retain(|i| !remove_set.contains(i));
            }
        }
        if frames.is_empty() {
            let placed = prepared.len() - remaining.len();
            return Err(TexPackerError::OutOfSpaceGeneric {
                placed,
                total: prepared.len(),
            });
        }

        let (page_w, page_h) = compute_page_size(&frames, &cfg);

        let page = Page {
            id: page_id,
            width: page_w,
            height: page_h,
            frames: frames.clone(),
        };
        atlas_pages.push(page);
        page_id += 1;
    }

    let meta = Meta {
        schema_version: "1".into(),
        app: "tex-packer".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        format: "RGBA8888".into(),
        scale: 1.0,
        power_of_two: cfg.power_of_two,
        square: cfg.square,
        max_dim: (cfg.max_width, cfg.max_height),
        padding: (cfg.border_padding, cfg.texture_padding),
        extrude: cfg.texture_extrusion,
        allow_rotation: cfg.allow_rotation,
        trim_mode: if cfg.trim { "trim" } else { "none" }.into(),
        background_color: None,
    };
    Ok(Atlas {
        pages: atlas_pages,
        meta,
    })
}

/// Compute final page dimensions given placed frames and config.
fn compute_page_size(frames: &[Frame], cfg: &PackerConfig) -> (u32, u32) {
    if cfg.force_max_dimensions {
        // When forced, return exactly the configured dimensions, ignoring pow2/square adjustments.
        return (cfg.max_width, cfg.max_height);
    }
    let pad_half = cfg.texture_padding / 2;
    let pad_rem = cfg.texture_padding - pad_half;
    let right_extra = cfg.texture_extrusion + pad_rem;
    let bottom_extra = cfg.texture_extrusion + pad_rem;
    let mut page_w = 0u32;
    let mut page_h = 0u32;
    for f in frames {
        page_w = page_w.max(f.frame.right() + 1 + right_extra + cfg.border_padding);
        page_h = page_h.max(f.frame.bottom() + 1 + bottom_extra + cfg.border_padding);
    }
    if cfg.power_of_two {
        page_w = next_pow2(page_w.max(1));
        page_h = next_pow2(page_h.max(1));
    }
    if cfg.square {
        let m = page_w.max(page_h);
        page_w = m;
        page_h = m;
    }
    (page_w, page_h)
}
