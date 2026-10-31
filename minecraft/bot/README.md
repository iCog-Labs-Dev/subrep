# SubRep Mineflayer bot

Install and start the bot with Node 22:

```bash
npm ci
SUBREP_BOT_TOKEN=test-token-123 node index.js
```

The bot connects to `127.0.0.1:25565` as `subrep_bot` using Minecraft
`1.21.8`. Its authenticated JSON API listens only on
`127.0.0.1:25566`. Every request must include `X-SubRep-Token` matching
the required `SUBREP_BOT_TOKEN` environment variable.

Endpoints:

- `POST /observe` returns position, health, food, and inventory.
- `POST /act` accepts `{"action":"wait"}`, `{"action":"move_hall"}`,
  or `{"action":"move_perimeter"}`.
- `POST /stop` disconnects the bot and exits.

Action gate:

| Actions | Response |
| --- | --- |
| `wait`, `move_hall`, `move_perimeter` | Working |
| `walk_route`, `sprint_route`, `deliver`, `craft_torches`, `place_torch`, `build_gate`, `build_golem`, `buy_book`, `cheap_trade`, `cure_villager`, `set_trade_hotkeys`, `shoot`, `eat` | HTTP 200 with `status: "not_implemented"` |
| Anything else, such as `fly` | HTTP 400 with `status: "error"` and `reason: "unknown_action"` |

The bot spawns at world spawn `(0, 64, 0)`. `move_hall` targets
`(40, 64, 0)`, the hall player start from `minecraft/world/areas.json`.
`move_perimeter` targets `(40, 64, 8)`, a clear interior point. Movement
waits up to 30 seconds. Requests are limited to 10 per second and bodies
are limited to 1 MB.
