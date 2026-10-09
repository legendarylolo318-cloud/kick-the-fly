//! The screens behind the pause menu's buttons, one module per `ui/*.py` file. [`install`] registers all of them on a menu.

use crate::menu::Menu;

pub mod loadout;
pub mod share;
pub mod whatsnew;

pub fn install(menu: &mut Menu) {
    menu.register_page("loadout", Box::<loadout::LoadoutPage>::default());
    menu.register_page("loadout_notice", Box::<loadout::LoadoutNotice>::default());
    menu.register_page("share", Box::<share::SharePage>::default());
    menu.register_page("whatsnew", Box::new(whatsnew::WhatsNew));
}
