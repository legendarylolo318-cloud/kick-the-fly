//! The screens behind the pause menu's buttons, one module per `ui/*.py` file. [`install`] registers all of them on a menu.

use crate::menu::Menu;

pub mod arcade;
pub mod challenges;
pub mod critical;
pub mod day4;
pub mod diag;
pub mod lab;
pub mod load_state;
pub mod loadout;
pub mod neurodex;
pub mod shapes_ask;
pub mod share;
pub mod whatsnew;
pub mod wiring;

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
    menu.register_page("challenges", Box::new(challenges::ChallengesPage));
    menu.register_page("selftest", Box::new(diag::SelfTestPage));
    menu.register_page("bugreport", Box::<diag::BugReportPage>::default());
    menu.register_page("arcade", Box::<arcade::ArcadePage>::default());
    menu.register_page("lab", Box::new(lab::LabHub));
    menu.register_page("lab_params", Box::new(lab::LabParams));
    menu.register_page("lab_assumptions", Box::new(lab::LabAssumptions));
    menu.register_page("lab_validation", Box::new(lab::LabValidation));
    menu.register_page("lab_assays", Box::<lab::LabAssays>::default());
    menu.register_page("lab_asymmetry", Box::new(lab::LabAsymmetry));
    menu.register_page("lab_benchmark", Box::new(lab::LabBenchmark));
    menu.register_page("lab_netsci", Box::<day4::LabNetsci>::default());
    menu.register_page("lab_sleepdep", Box::<day4::LabSleepdep>::default());
    menu.register_page("lab_sensitivity", Box::<day4::LabSensitivity>::default());
    menu.register_page("lab_wiring", Box::<wiring::LabWiring>::default());
    menu.register_page("lab_critical", Box::<critical::LabCritical>::default());
    menu.register_page("whatsnew", Box::new(whatsnew::WhatsNew));
}
