# dusk_v1 village build

## Simulator facts

The source simulator start state is `dusk_state(VillageMission())`:

- Mission: `village_defense_trade_v1`.
- Dusk is tick `12000`; sunrise is tick `23000`.
- Four protected villagers start at 20 health; two are librarians.
- One zombie villager starts in the state.
- The player starts at hall with health 20, food 14, and armor 4.
- The wall starts at `wall_fraction=0.6`, interpreted here as a perimeter
  with gaps. Lighting starts at `lit_fraction=0.3`.
- Trading access starts open.
- The start state has `golems=0`; no golem is summoned.
- The simulator chest/delivery state starts empty. No chest contents are
  invented here.
- The simulator start inventory is 20 emeralds, 4 torches, 2 charcoal,
  4 coal, 4 iron blocks, 1 pumpkin, 3 food items, and 1 cure kit.

## Coordinates

These coordinates are decisions for the real server build, not coordinates
specified by the simulator.

- Spawn: `(0, 64, 0)`.
- Hall bounds: `(38, 63, -3)` through `(42, 68, 3)`.
- Hall/player start: `(40, 64, 0)`.
- Perimeter bounds: `(32, 63, -10)` through `(48, 68, 10)`.
- Settlement bounds: `(52, 63, -4)` through `(58, 68, 4)`.
- Delivery chest: `(55, 64, 0)`.
- Route waypoints: `(44, 63, 0)`, `(48, 63, 0)`, `(52, 63, 0)`.
- The nearest build coordinate is 32 blocks from spawn, outside the
  unchanged 16-block spawn-protection radius.

## Build decisions

- The build uses stone, stone bricks, oak planks, dirt paths, lanterns,
  torches, and iron bars because the simulator does not specify materials.
- The perimeter is a 17-by-21 rectangle, four blocks high, with 28 gap
  columns at y=64 in 8 sections: west/east gaps are 4 wide and
  south/north gaps are 3 wide. This is 44 solid columns of 72, or 0.61
  coverage before the route gap.
- The route adds 3 more gap columns at `(48, 64, -1)`, `(48, 64, 0)`, and
  `(48, 64, 1)`, making 31 gap columns and 41/72 = 0.57 coverage, still
  approximately the simulator's 0.6 wall fraction.
- The interior is cleared from `(33, 64, -9)` through `(47, 67, 9)`, so
  the perimeter remains one block thick and the village interior is open.
- Six perimeter torch positions differ from `gap_positions`: five perimeter
  torches are inside gap columns, and one wall torch punches a one-block
  hole. One additional torch is in the hall. This deterministic layout is
  the chosen concrete representation of the simulator's 0.3 lit fraction;
  the digest remains valid.
- The two unspecified protected villager professions are farmers. The two
  librarians are at `(39, 64, 0)` and `(41, 64, 0)`.
- The caged zombie villager is at `(35, 64, -6)`, with an iron-bars cage.
- The settlement chest is empty because the simulator starts with zero
  books delivered and specifies no world chest contents.
- The hall is an open floor-and-roof shelter inside the cleared perimeter;
  it has no walls. The settlement is a full-height-walled rectangle with a
  west-wall door gap at `(52, 64, 0)`. The route is a straight dirt-path
  segment through the listed waypoints.
- The build keeps the site force-loaded with `/forceload add 1 -1 4 1`;
  these are chunk coordinates covering the block-coordinate build area.
- The build commands use `/tick freeze` after setting dusk. This freezes
  simulation time for the snapshot; unfreeze with `/tick unfreeze` when a
  live run is intended.

## Reproducible build and save procedure

The command file is a manual server-console script. Lines beginning with
`/` are Minecraft commands and must be entered one at a time in the Paper
console, in their existing order. Headings, blank lines, and the terminal
commands are instructions for the operator, not Minecraft commands. Wait
for each command's response before entering the next command.

1. If Paper is running, enter `stop` in its server console and wait until
   saving has completed and the process exits.
2. From a terminal, change to `minecraft/server`. To preserve an existing
   world before a deliberate rebuild, make a copy with
   `cp -a world "world_backup_$(date +%Y%m%d_%H%M%S)"`. Backup directories
   match the repository ignore rule and are not part of the build.
3. Remove only the active `world/` directory with `rm -rf world`, then
   start Paper with `./start.sh`. This creates a fresh default world.
4. After the console prints `Done (...)! For help, type "help"`, enter the
   slash commands from `dusk_v1_build_commands.txt` one at a time. The
   commands set the spawn, force-load the site, flatten the area, build the
   structures, add the entities and empty chest, clean up stray entities,
   set time to tick `12000`, and freeze the simulation.
5. Confirm the verification messages in the file appear. In particular,
   `interior_ok`, `route_gap_ok`, `door_ok`, `wall_ok`, and
   `settlement_wall_ok` must be printed by their corresponding checks.
   To repeat those checks after restarting without rebuilding, enter:

   ```text
   /execute if block 44 65 0 minecraft:air run say interior_ok
   /execute if block 48 65 0 minecraft:air run say route_gap_ok
   /execute if block 52 65 0 minecraft:air run say door_ok
   /execute if block 32 65 0 minecraft:stone_bricks run say wall_ok
   /execute if block 55 66 -4 minecraft:oak_planks run say settlement_wall_ok
   ```

   A successful check prints its label. No label means that condition
   failed and the world should not be treated as verified.
6. Enter `/save-all` and wait for `Saved the game`. This writes the
   frozen dusk snapshot to `minecraft/server/world/`.
7. Enter `stop` and wait for Paper to finish saving and exit. The saved
   `world/` directory is the snapshot to preserve; do not remove it unless
   rebuilding.
8. To inspect the saved snapshot later, run `./start.sh` without deleting
   `world/`, wait for the `Done (...)!` line, and enter the five checks
   listed above. Enter `/save-all` before a clean `stop` when any changes
   were made.

The build was executed with this procedure and the five final checks all
passed. The active snapshot is the `world/` directory; timestamped
`world_backup_*` directories are optional rollback copies, not alternate
snapshots used by Paper.

## Entities

| Entity | Position | Details |
|---|---:|---|
| Villager | `(39, 64, 0)` | Librarian, level 2, plains |
| Villager | `(41, 64, 0)` | Librarian, level 2, plains |
| Villager | `(39, 64, 2)` | Farmer, level 2, plains; build decision |
| Villager | `(41, 64, 2)` | Farmer, level 2, plains; build decision |
| Zombie villager | `(35, 64, -6)` | Farmer, level 1, plains, inside iron-bars cage |

No iron golem is present because the simulator start state has zero golems.

Five perimeter torch positions are inside gap columns (including the route gap at (48,64,0)), one wall torch punches a one-block hole at (32,64,0), and the door gap at (52,64,0) finally holds a lantern; one additional torch is in the hall.

## Open bot item

Starting inventory is set when the bot first joins (console `/give` + save),
before the snapshot is frozen.
