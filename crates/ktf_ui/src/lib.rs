//! The menus of Kick the Fly, ported from `legacy-python/kickthefly/ui/menu.py` and drawn the way pygame drew them: an immediate-mode
//! page function per screen paints rectangles, circles and text onto a [`canvas::Canvas`] and registers hit regions; the next mouse or
//! key event is matched against them. Same colors, same pixel sizes, same labels, same order of events.
//!
//! The crate has no Bevy in it. The game shows the canvas as a full-window image and feeds it input; tests and the
//! `render_menu` example draw pages to PNG files, which is how the layout is compared with screenshots of the pygame menu
//! (`legacy-python/make_ui_reference.py`).

// the widget functions keep the argument lists of the pygame originals (rect, label, click, id, style, enabled, tip, ...)
#![allow(clippy::too_many_arguments)]

pub mod canvas;
pub mod font;
pub mod i18n;
pub mod menu;
pub mod theme;

pub use canvas::{Canvas, Rect};
pub use menu::{Effect, Key, Menu, UiEvent};
