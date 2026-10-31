const http = require("http");
const mineflayer = require("mineflayer");
const { pathfinder, Movements, goals } = require("mineflayer-pathfinder");

const HOST = "127.0.0.1";
const PORT = 25566;
const TOKEN = process.env.SUBREP_BOT_TOKEN;
const MAX_BODY_BYTES = 1024 * 1024;
const RATE_WINDOW_MS = 1000;
const RATE_LIMIT = 10;
const NOT_IMPLEMENTED_ACTIONS = new Set([
  "walk_route",
  "sprint_route",
  "deliver",
  "craft_torches",
  "place_torch",
  "build_gate",
  "build_golem",
  "buy_book",
  "cheap_trade",
  "cure_villager",
  "set_trade_hotkeys",
  "shoot",
  "eat",
]);

if (!TOKEN) {
  console.error("SUBREP_BOT_TOKEN is required");
  process.exit(1);
}

const bot = mineflayer.createBot({
  host: "127.0.0.1",
  port: 25565,
  username: "subrep_bot",
  version: "1.21.8",
});
bot.loadPlugin(pathfinder);

let movements;
let stopping = false;
const requestTimes = [];

function observation() {
  if (!bot.entity) {
    throw new Error("bot is not spawned");
  }
  const position = bot.entity.position;
  return {
    status: "ok",
    position: [Math.floor(position.x), Math.floor(position.y), Math.floor(position.z)],
    health: bot.health,
    food: bot.food,
    inventory: bot.inventory.items().map((item) => ({
      id: item.name,
      count: item.count,
    })),
  };
}

function sendJson(response, statusCode, payload) {
  const body = JSON.stringify(payload);
  response.writeHead(statusCode, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  response.end(body);
}

function authorized(request) {
  return request.headers["x-subrep-token"] === TOKEN;
}

function rateAllowed() {
  const now = Date.now();
  while (requestTimes.length && requestTimes[0] <= now - RATE_WINDOW_MS) {
    requestTimes.shift();
  }
  if (requestTimes.length >= RATE_LIMIT) {
    return false;
  }
  requestTimes.push(now);
  return true;
}

function readBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    request.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(Object.assign(new Error("request body too large"), { statusCode: 413 }));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}"));
      } catch {
        reject(Object.assign(new Error("invalid_json"), { statusCode: 400 }));
      }
    });
    request.on("error", reject);
  });
}

function moveTo(x, y, z) {
  if (!movements) {
    throw new Error("bot is not ready");
  }
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      cleanup();
      resolve({ status: "error", reason: "movement_timeout" });
    }, 30000);
    const onGoalReached = () => {
      cleanup();
      resolve(observation());
    };
    const onPathUpdate = (result) => {
      if (result.status === "noPath") {
        cleanup();
        resolve({ status: "error", reason: "no_path" });
      }
    };
    function cleanup() {
      clearTimeout(timeout);
      bot.removeListener("goal_reached", onGoalReached);
      bot.removeListener("path_update", onPathUpdate);
      bot.pathfinder.stop();
    }
    bot.once("goal_reached", onGoalReached);
    bot.on("path_update", onPathUpdate);
    bot.pathfinder.setMovements(movements);
    bot.pathfinder.setGoal(new goals.GoalBlock(x, y, z));
  });
}

async function handle(request, response) {
  if (!authorized(request)) {
    sendJson(response, 401, { status: "error", reason: "unauthorized" });
    return;
  }
  if (!rateAllowed()) {
    sendJson(response, 429, { status: "error", reason: "rate_limited" });
    return;
  }
  if (request.method !== "POST" || !["/observe", "/act", "/stop"].includes(request.url)) {
    sendJson(response, 404, { status: "error", reason: "not_found" });
    return;
  }
  let body;
  try {
    body = await readBody(request);
  } catch (error) {
    sendJson(response, error.statusCode || 400, {
      status: "error",
      reason: error.statusCode === 413 ? "body_too_large" : error.message,
    });
    return;
  }
  try {
    if (request.url === "/observe") {
      sendJson(response, 200, observation());
    } else if (request.url === "/act") {
      if (NOT_IMPLEMENTED_ACTIONS.has(body.action)) {
        sendJson(response, 200, {
          status: "not_implemented",
          action: body.action,
        });
      } else if (body.action === "wait") {
        sendJson(response, 200, observation());
      } else if (body.action === "move_hall") {
        sendJson(response, 200, await moveTo(40, 64, 0));
      } else if (body.action === "move_perimeter") {
        sendJson(response, 200, await moveTo(40, 64, 8));
      } else {
        sendJson(response, 400, { status: "error", reason: "unknown_action" });
      }
    } else {
      sendJson(response, 200, { status: "ok" });
      stopping = true;
      bot.quit("emergency stop");
      setImmediate(() => process.exit(0));
    }
  } catch (error) {
    sendJson(response, 503, { status: "error", reason: error.message });
  }
}

const server = http.createServer((request, response) => {
  handle(request, response).catch((error) => {
    if (!response.headersSent) {
      sendJson(response, 500, { status: "error", reason: error.message });
    }
  });
});

server.listen(PORT, HOST, () => {
  console.log(`Sidecar listening on http://${HOST}:${PORT}`);
});

bot.once("spawn", () => {
  movements = new Movements(bot);
  console.log("Bot spawned at world spawn");
});
bot.on("end", () => {
  if (stopping) {
    return;
  }
  console.error("Bot disconnected from the server");
  process.exit(1);
});
bot.on("error", (error) => {
  console.error(`Bot error: ${error.message}`);
});
