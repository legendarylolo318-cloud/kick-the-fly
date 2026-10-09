//! The screens behind the pause menu's buttons, one module per `ui/*.py` file. [`install`] registers all of them on a menu.

use crate::menu::Menu;

pub mod diag;
pub mod load_state;
pub mod loadout;
pub mod neurodex;
pub mod shapes_ask;
pub mod share;
pub mod whatsnew;

pub fn install(menu: &mut Menu) {
    install_with(menu, std::sync::Arc::new(std::sync::Mutex::new(neurodex::NeurodexShared::new(Default::default()))));
}

/// Register every page; `dex` is the state the Neurodex panel shares with the game.
pub fn install_with(menu: &mut Menu, dex: neurodex::SharedDex) {
    menu.register_page("neurodex", Box::new(neurodex::NeurodexPage::new(dex)));
    menu.register_page("load_state", Box::new(load_state::LoadState));
    menu.register_page("loadout", Box::<loadout::LoadoutPage>::default());
    menu.register_page("loadout_notice", Box::<loadout::LoadoutNotice>::default());
    menu.register_page("neuron_shapes_ask", Box::new(shapes_ask::ShapesAsk));
    menu.register_page("share", Box::<share::SharePage>::default());
    menu.register_page("selftest", Box::new(diag::SelfTestPage));
    menu.register_page("bugreport", Box::<diag::BugReportPage>::default());
    menu.register_page("whatsnew", Box::new(whatsnew::WhatsNew));
}
