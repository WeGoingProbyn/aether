mod plugins;
mod resources;

use bevy::prelude::*;
use plugins::simulation::SimulationPlugin;
use plugins::rendering::RenderingPlugin;


fn main() {
  App::new()
    .add_plugins(DefaultPlugins.set(WindowPlugin {
      primary_window: Some(Window {
        title: "Aether".into(),
        resizable: true,
        ..default()
      }),
      ..default()
    }))
    .add_plugins((SimulationPlugin, RenderingPlugin))
    .run();
}
