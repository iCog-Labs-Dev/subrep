const http = require("http");
const fs = require("fs");
const mineflayer = require("mineflayer");
const path = require("path");
const { Vec3 } = require("vec3");
const { pathfinder, Movements, goals } = require("mineflayer-pathfinder");

const HOST = "127.0.0.1";
const PORT = 25566;
const TOKEN = process.env.SUBREP_BOT_TOKEN;
const MAX_BODY_BYTES = 1024 * 1024;
const RATE_WINDOW_MS = 1000;
const RATE_LIMIT = 10;
const RUN_TOKEN_EXPIRES_AT = Number(process.env.SUBREP_BOT_TOKEN_EXPIRES_AT || 0) || Date.now() + 3600000;
const AREAS_PATH = path.resolve(__dirname, "../world/areas.json");

function loadRouteWaypoints() {
  let areas;
  try {
    areas = JSON.parse(fs.readFileSync(AREAS_PATH, "utf8"));
  } catch (error) {
    console.error(`Unable to load route waypoints from ${AREAS_PATH}: ${error.message}`);
    process.exit(1);
  }
  const waypoints = areas && areas.route && areas.route.waypoints;
  if (
    !Array.isArray(waypoints)
    || waypoints.length === 0
    || waypoints.some((position) => (
      !Array.isArray(position)
      || position.length !== 3
      || position.some((coordinate) => !Number.isInteger(coordinate))
    ))
  ) {
    console.error(`Invalid route.waypoints in ${AREAS_PATH}`);
    process.exit(1);
  }
  const torchPositions = areas && areas.torch_positions;
  if (
    !Array.isArray(torchPositions)
    || torchPositions.length === 0
    || torchPositions.some((position) => (
      !Array.isArray(position)
      || position.length !== 3
      || position.some((coordinate) => !Number.isInteger(coordinate))
    ))
  ) {
    console.error(`Invalid torch_positions in ${AREAS_PATH}`);
    process.exit(1);
  }
  const gapPositions = areas && areas.perimeter && areas.perimeter.gap_positions;
  if (
    !Array.isArray(gapPositions)
    || gapPositions.some((position) => (
      !Array.isArray(position)
      || position.length !== 3
      || position.some((coordinate) => !Number.isInteger(coordinate))
    ))
  ) {
    console.error(`Invalid perimeter.gap_positions in ${AREAS_PATH}`);
    process.exit(1);
  }
  const golemSpot = areas && areas.golem_spot;
  if (
    !Array.isArray(golemSpot)
    || golemSpot.length !== 3
    || golemSpot.some((coordinate) => !Number.isInteger(coordinate))
  ) {
    console.error(`Invalid golem_spot in ${AREAS_PATH}`);
    process.exit(1);
  }
  const deliveryChest = areas && areas.settlement && areas.settlement.delivery_chest;
  if (
    !Array.isArray(deliveryChest)
    || deliveryChest.length !== 3
    || deliveryChest.some((coordinate) => !Number.isInteger(coordinate))
  ) {
    console.error(`Invalid settlement.delivery_chest in ${AREAS_PATH}`);
    process.exit(1);
  }
  return { waypoints, torchPositions, gapPositions, golemSpot, deliveryChest };
}

const AREAS_DATA = loadRouteWaypoints();
const ROUTE_WAYPOINTS = AREAS_DATA.waypoints;
const TORCH_POSITIONS = AREAS_DATA.torchPositions;
const GAP_POSITIONS = AREAS_DATA.gapPositions;
const GOLEM_SPOT = AREAS_DATA.golemSpot;
const DELIVERY_CHEST = AREAS_DATA.deliveryChest;
const LIBRARIAN_POSITIONS = [[68, 64, -2], [70, 64, -2]];
const ZOMBIE_POSITION = [67, 64, 8];
const NOT_IMPLEMENTED_ACTIONS = new Set([
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
const seenMessageIds = new Set();
let golemIronPlaced = false;
let pendingBookTrade = null;
let curePotionThrown = false;

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
  return request.headers["x-subrep-token"] === TOKEN && Date.now() < RUN_TOKEN_EXPIRES_AT;
}

function envelope(payload, body) {
  if (!body || typeof body.message_id !== "string") {
    return payload;
  }
  return Object.assign(
    {
      message_id: body.message_id,
      snapshot_id: body.snapshot_id || null,
      world_tick: body.world_tick == null ? null : body.world_tick,
    },
    payload,
  );
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
    let tooLarge = false;
    request.on("data", (chunk) => {
      if (tooLarge) {
        return;
      }
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        tooLarge = true;
        reject(Object.assign(new Error("request body too large"), { statusCode: 413 }));
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => {
      if (tooLarge) {
        return;
      }
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
  const position = bot.entity.position;
  if (
    Math.floor(position.x) === x
    && Math.floor(position.y) === y
    && Math.floor(position.z) === z
  ) {
    return observation();
  }
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

function nextRouteWaypoint() {
  const currentX = Math.floor(bot.entity.position.x);
  return ROUTE_WAYPOINTS.find((position) => position[0] > currentX);
}

async function moveRoute(sprint) {
  const waypoint = nextRouteWaypoint();
  if (!waypoint) {
    return observation();
  }
  if (sprint) {
    bot.setControlState("sprint", true);
  }
  try {
    return await moveTo(waypoint[0], waypoint[1], waypoint[2]);
  } finally {
    if (sprint) {
      bot.setControlState("sprint", false);
    }
  }
}

function inventoryCount(name) {
  return bot.inventory.items()
    .filter((item) => item.name === name)
    .reduce((total, item) => total + item.count, 0);
}

function blockPosition(x, y, z) {
  return new Vec3(x, y, z);
}

function moveNear(x, y, z, radius) {
  const target = blockPosition(x, y, z);
  if (bot.entity.position.distanceTo(target) <= radius) {
    return observation();
  }
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
    bot.pathfinder.setGoal(new goals.GoalNear(x, y, z, radius));
  });
}

async function craftTorches() {
  if (inventoryCount("stick") < 1) {
    return { status: "error", reason: "missing_stick" };
  }
  const fuelName = inventoryCount("charcoal") > 0 ? "charcoal" : "coal";
  if (inventoryCount(fuelName) < 1) {
    return { status: "error", reason: "missing_fuel" };
  }
  const torchId = bot.registry.itemsByName.torch.id;
  const fuelId = bot.registry.itemsByName[fuelName].id;
  const recipes = bot.recipesFor(torchId, null, 1, null);
  const recipe = recipes.find((candidate) => (
    (candidate.ingredients || []).some((ingredient) => (
      ingredient.id === fuelId
      && (fuelName === "charcoal" ? ingredient.metadata === 1 : ingredient.metadata !== 1)
    )) || (candidate.inShape || []).some((row) => row.some((ingredient) => (
      ingredient.id === fuelId
      && (fuelName === "charcoal" ? ingredient.metadata === 1 : ingredient.metadata !== 1)
    )))
  )) || recipes[0];
  if (!recipe) {
    return { status: "error", reason: "no_torch_recipe" };
  }
  await bot.craft(recipe, 1, null);
  return observation();
}

async function placeTorch() {
  if (inventoryCount("torch") < 1) {
    return { status: "error", reason: "missing_torch" };
  }
  const current = bot.entity.position;
  const candidates = TORCH_POSITIONS
    .map((position) => ({
      position,
      distance: current.distanceTo(blockPosition(position[0], position[1], position[2])),
    }))
    .sort((left, right) => left.distance - right.distance);
  for (const candidate of candidates) {
    const [x, y, z] = candidate.position;
    const target = bot.blockAt(blockPosition(x, y, z));
    const support = bot.blockAt(blockPosition(x, y - 1, z));
    if (
      !target
      || target.name !== "air"
      || !support
      || support.boundingBox !== "block"
    ) {
      continue;
    }
    const movement = await moveNear(x, y, z, 3);
    if (movement.status !== "ok") {
      continue;
    }
    const placedTarget = bot.blockAt(blockPosition(x, y, z));
    const placedSupport = bot.blockAt(blockPosition(x, y - 1, z));
    if (
      !placedTarget
      || placedTarget.name !== "air"
      || !placedSupport
      || placedSupport.boundingBox !== "block"
    ) {
      continue;
    }
    const torch = bot.inventory.items().find((item) => item.name === "torch");
    if (!torch) {
      return { status: "error", reason: "missing_torch" };
    }
    try {
      await bot.equip(torch, "hand");
      await bot.placeBlock(placedSupport, new Vec3(0, 1, 0));
      return observation();
    } catch (error) {
      console.error(
        `Unable to place torch at (${x}, ${y}, ${z}) on ${placedSupport.name}: ${error.message}`,
      );
    }
  }
  return { status: "error", reason: "no_placeable_torch_position" };
}

async function buildGate() {
  if (inventoryCount("oak_fence_gate") < 1) {
    return { status: "error", reason: "missing_gate" };
  }
  const current = bot.entity.position;
  const candidates = GAP_POSITIONS
    .map((position) => ({
      position,
      distance: current.distanceTo(blockPosition(position[0], position[1], position[2])),
    }))
    .sort((left, right) => left.distance - right.distance);
  for (const candidate of candidates) {
    const [x, y, z] = candidate.position;
    const target = bot.blockAt(blockPosition(x, y, z));
    const support = bot.blockAt(blockPosition(x, y - 1, z));
    if (!target || target.name !== "air" || !support || support.boundingBox !== "block") {
      continue;
    }
    const movement = await moveNear(x, y, z, 3);
    if (movement.status !== "ok") {
      continue;
    }
    const placedTarget = bot.blockAt(blockPosition(x, y, z));
    const placedSupport = bot.blockAt(blockPosition(x, y - 1, z));
    if (!placedTarget || placedTarget.name !== "air" || !placedSupport || placedSupport.boundingBox !== "block") {
      continue;
    }
    const gate = bot.inventory.items().find((item) => item.name === "oak_fence_gate");
    if (!gate) {
      return { status: "error", reason: "missing_gate" };
    }
    try {
      await bot.equip(gate, "hand");
      await bot.placeBlock(placedSupport, new Vec3(0, 1, 0));
      return observation();
    } catch (error) {
      console.error(`Unable to place gate at (${x}, ${y}, ${z}) on ${placedSupport.name}: ${error.message}`);
    }
  }
  return { status: "error", reason: "no_placeable_gate_position" };
}

async function buildGolem() {
  const [x, y, z] = GOLEM_SPOT;
  if (!golemIronPlaced) {
    if (inventoryCount("iron_block") < 4) {
      return { status: "error", reason: "missing_iron" };
    }
    const hallMovement = await moveTo(40, 64, 0);
    if (hallMovement.status !== "ok") {
      return hallMovement;
    }
    const clearMovement = await moveNear(x, y, z - 1, 1);
    if (clearMovement.status !== "ok") {
      return clearMovement;
    }
    const iron = bot.inventory.items().find((item) => item.name === "iron_block");
    if (!iron) {
      return { status: "error", reason: "missing_iron" };
    }
    await bot.equip(iron, "hand");
    const base = bot.blockAt(blockPosition(x, y - 1, z));
    const middle = bot.blockAt(blockPosition(x, y, z));
    if (!base || base.boundingBox !== "block" || !middle || middle.name !== "air") {
      return { status: "error", reason: "no_placeable_golem_position" };
    }
    await bot.placeBlock(base, new Vec3(0, 1, 0));
    const baseBlock = bot.blockAt(blockPosition(x, y, z));
    if (!baseBlock) {
      return { status: "error", reason: "no_placeable_golem_position" };
    }
    await bot.placeBlock(baseBlock, new Vec3(0, 1, 0));
    const middleBlock = bot.blockAt(blockPosition(x, y + 1, z));
    if (!middleBlock) {
      return { status: "error", reason: "no_placeable_golem_position" };
    }
    const leftMovement = await moveNear(x - 2, y, z, 3);
    if (leftMovement.status !== "ok") {
      return leftMovement;
    }
    await bot.placeBlock(middleBlock, new Vec3(-1, 0, 0));
    const rightMovement = await moveNear(x + 2, y, z, 3);
    if (rightMovement.status !== "ok") {
      return rightMovement;
    }
    await bot.placeBlock(middleBlock, new Vec3(1, 0, 0));
    golemIronPlaced = true;
    return observation();
  }
  if (inventoryCount("carved_pumpkin") < 1) {
    return { status: "error", reason: "missing_pumpkin" };
  }
  const hallMovement = await moveTo(40, 64, 0);
  if (hallMovement.status !== "ok") {
    return hallMovement;
  }
  const clearMovement = await moveNear(x, y, z - 1, 1);
  if (clearMovement.status !== "ok") {
    return clearMovement;
  }
  const pumpkin = bot.inventory.items().find((item) => item.name === "carved_pumpkin");
  const middle = bot.blockAt(blockPosition(x, y + 1, z));
  if (!pumpkin || !middle) {
    return { status: "error", reason: "missing_pumpkin" };
  }
  try {
    await bot.equip(pumpkin, "hand");
    await bot.placeBlock(middle, new Vec3(0, 1, 0));
  } catch (error) {
    console.error(`Unable to place carved pumpkin at (${x}, ${y + 2}, ${z}): ${error.message}`);
    await new Promise((resolve) => setTimeout(resolve, 250));
    const pumpkinPlaced = bot.blockAt(blockPosition(x, y + 2, z));
    const golemSpawned = Object.values(bot.entities).some((entity) => entity.name === "iron_golem");
    if (!pumpkinPlaced || pumpkinPlaced.name !== "carved_pumpkin") {
      if (!golemSpawned) {
        return { status: "error", reason: "no_placeable_golem_position" };
      }
    }
  }
  golemIronPlaced = false;
  return observation();
}

function findLibrarian(position) {
  const target = blockPosition(position[0], position[1], position[2]);
  return Object.values(bot.entities).find((entity) => (
    entity.name === "villager"
    && entity.position.distanceTo(target) <= 2
  ));
}

async function openLibrarianTrade(predicate) {
  for (const position of LIBRARIAN_POSITIONS) {
    const librarian = findLibrarian(position);
    if (!librarian) {
      continue;
    }
    const movement = await moveNear(position[0], position[1], position[2], 3);
    if (movement.status !== "ok") {
      continue;
    }
    const window = await bot.openVillager(librarian);
    const tradeIndex = window.trades.findIndex(predicate);
    if (tradeIndex >= 0) {
      return { window, trade: window.trades[tradeIndex], tradeIndex };
    }
    window.close();
  }
  return null;
}

async function buyBook() {
  if (!pendingBookTrade) {
    const found = await openLibrarianTrade((trade) => (
      trade.outputs.some((item) => item.name === "book")
      && trade.inputs.some((item) => item.name === "emerald")
      && trade.realPrice === 7
    ));
    if (!found) {
      return { status: "error", reason: "no_book_trade" };
    }
    const price = found.trade.realPrice;
    pendingBookTrade = {
      librarian: found.window,
      tradeIndex: found.tradeIndex,
      price,
    };
    return Object.assign(observation(), { book_price: price });
  }
  const pending = pendingBookTrade;
  pendingBookTrade = null;
  if (inventoryCount("emerald") < pending.price) {
    pending.librarian.close();
    return { status: "error", reason: "not_enough_emeralds" };
  }
  try {
    await pending.librarian.trade(pending.tradeIndex, 1);
    pending.librarian.close();
    return observation();
  } catch (error) {
    pending.librarian.close();
    return { status: "error", reason: `book_trade_failed: ${error.message}` };
  }
}

async function cheapTrade() {
  if (inventoryCount("emerald") < 1) {
    return { status: "error", reason: "missing_emerald" };
  }
  const found = await openLibrarianTrade((trade) => (
    trade.inputs.some((item) => item.name === "emerald")
    && trade.realPrice === 1
  ));
  if (!found) {
    return { status: "error", reason: "no_cheap_trade" };
  }
  try {
    await found.window.trade(found.tradeIndex, 1);
    found.window.close();
    return observation();
  } catch (error) {
    found.window.close();
    return { status: "error", reason: `cheap_trade_failed: ${error.message}` };
  }
}

function findZombieVillager() {
  const target = blockPosition(ZOMBIE_POSITION[0], ZOMBIE_POSITION[1], ZOMBIE_POSITION[2]);
  return Object.values(bot.entities).find((entity) => (
    entity.name === "zombie_villager"
    && entity.position.distanceTo(target) <= 2
  ));
}

async function cureVillager() {
  if (!curePotionThrown) {
    if (inventoryCount("splash_potion") < 1) {
      return { status: "error", reason: "missing_potion" };
    }
    const movement = await moveNear(ZOMBIE_POSITION[0], ZOMBIE_POSITION[1], ZOMBIE_POSITION[2], 3);
    if (movement.status !== "ok") {
      return movement;
    }
    const zombie = findZombieVillager();
    const potion = bot.inventory.items().find((item) => item.name === "splash_potion");
    if (!zombie || !potion) {
      return { status: "error", reason: "no_zombie_villager" };
    }
    try {
      await bot.equip(potion, "hand");
      await bot.lookAt(zombie.position.offset(0, 1, 0), true);
      bot.activateItem();
      curePotionThrown = true;
      return observation();
    } catch (error) {
      return { status: "error", reason: `potion_throw_failed: ${error.message}` };
    }
  }
  if (inventoryCount("golden_apple") < 1) {
    return { status: "error", reason: "missing_apple" };
  }
  const movement = await moveNear(ZOMBIE_POSITION[0], ZOMBIE_POSITION[1], ZOMBIE_POSITION[2], 3);
  if (movement.status !== "ok") {
    return movement;
  }
  const zombie = findZombieVillager();
  const apple = bot.inventory.items().find((item) => item.name === "golden_apple");
  if (!zombie || !apple) {
    return { status: "error", reason: "no_zombie_villager" };
  }
  try {
    await bot.equip(apple, "hand");
    await bot.activateEntity(zombie);
    curePotionThrown = false;
    return observation();
  } catch (error) {
    return { status: "error", reason: `apple_feed_failed: ${error.message}` };
  }
}

function nearestHostile() {
  return bot.nearestEntity((entity) => (
    ["zombie", "skeleton", "spider", "creeper", "witch", "phantom"].includes(entity.name)
    && bot.entity.position.distanceTo(entity.position) <= 30
  ));
}

async function shoot() {
  if (inventoryCount("bow") < 1) {
    return { status: "error", reason: "missing_bow" };
  }
  if (inventoryCount("arrow") < 1) {
    return { status: "error", reason: "missing_arrow" };
  }
  const bow = bot.inventory.items().find((item) => item.name === "bow");
  const target = nearestHostile();
  const aim = target
    ? target.position.offset(0, target.height || 1, 0)
    : bot.entity.position.offset(-Math.sin(bot.entity.yaw) * 30, 1.5, Math.cos(bot.entity.yaw) * 30);
  try {
    await bot.equip(bow, "hand");
    await bot.lookAt(aim, true);
    bot.activateItem();
    await new Promise((resolve) => setTimeout(resolve, 1000));
    bot.deactivateItem();
    return observation();
  } catch (error) {
    bot.deactivateItem();
    return { status: "error", reason: `shoot_failed: ${error.message}` };
  }
}

async function eat() {
  const food = bot.inventory.items().find((item) => (
    item.name === "cooked_beef" || item.name === "cooked_chicken"
  ));
  if (!food) {
    return { status: "error", reason: "missing_food" };
  }
  try {
    await bot.equip(food, "hand");
    bot.activateItem();
    await new Promise((resolve) => setTimeout(resolve, 1800));
    bot.deactivateItem();
    return observation();
  } catch (error) {
    bot.deactivateItem();
    return { status: "error", reason: `eat_failed: ${error.message}` };
  }
}

async function deliver() {
  if (inventoryCount("book") < 1) {
    return { status: "error", reason: "missing_book" };
  }
  const movement = await moveNear(DELIVERY_CHEST[0], DELIVERY_CHEST[1], DELIVERY_CHEST[2], 3);
  if (movement.status !== "ok") {
    return movement;
  }
  const chestBlock = bot.blockAt(blockPosition(
    DELIVERY_CHEST[0],
    DELIVERY_CHEST[1],
    DELIVERY_CHEST[2],
  ));
  if (!chestBlock || chestBlock.name !== "chest") {
    return { status: "error", reason: "delivery_chest_not_found" };
  }
  let chest;
  try {
    chest = await bot.openChest(chestBlock);
    for (const book of bot.inventory.items().filter((item) => item.name === "book")) {
      await chest.deposit(book.type, book.metadata, book.count);
    }
    chest.close();
    return observation();
  } catch (error) {
    if (chest) {
      chest.close();
    }
    return { status: "error", reason: `delivery_failed: ${error.message}` };
  }
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
  if (request.method === "GET" && ["/health", "/status"].includes(request.url)) {
    sendJson(response, 200, {
      status: "ok",
      health: bot.entity ? bot.health : null,
      spawned: Boolean(bot.entity),
      auth: "offline",
      token_expires_at: new Date(RUN_TOKEN_EXPIRES_AT).toISOString(),
    });
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
    if (body.message_id) {
      if (seenMessageIds.has(body.message_id)) {
        sendJson(response, 409, { status: "error", reason: "replay" });
        return;
      }
      seenMessageIds.add(body.message_id);
    }
    if (request.url === "/observe") {
      sendJson(response, 200, envelope(observation(), body));
    } else if (request.url === "/act") {
      if (NOT_IMPLEMENTED_ACTIONS.has(body.action)) {
        sendJson(response, 200, envelope({
          status: "not_implemented",
          action: body.action,
        }, body));
      } else if (body.action === "wait") {
        sendJson(response, 200, envelope(observation(), body));
      } else if (body.action === "move_hall") {
        sendJson(response, 200, envelope(await moveTo(40, 64, 0), body));
      } else if (body.action === "move_perimeter") {
        sendJson(response, 200, envelope(await moveTo(68, 64, 1), body));
      } else if (body.action === "walk_route") {
        sendJson(response, 200, envelope(await moveRoute(false), body));
      } else if (body.action === "sprint_route") {
        sendJson(response, 200, envelope(await moveRoute(true), body));
      } else if (body.action === "craft_torches") {
        sendJson(response, 200, envelope(await craftTorches(), body));
      } else if (body.action === "place_torch") {
        sendJson(response, 200, envelope(await placeTorch(), body));
      } else if (body.action === "build_gate") {
        sendJson(response, 200, envelope(await buildGate(), body));
      } else if (body.action === "build_golem") {
        sendJson(response, 200, envelope(await buildGolem(), body));
      } else if (body.action === "buy_book") {
        sendJson(response, 200, envelope(await buyBook(), body));
      } else if (body.action === "cheap_trade") {
        sendJson(response, 200, envelope(await cheapTrade(), body));
      } else if (body.action === "set_trade_hotkeys") {
        sendJson(response, 200, envelope(observation(), body));
      } else if (body.action === "cure_villager") {
        sendJson(response, 200, envelope(await cureVillager(), body));
      } else if (body.action === "shoot") {
        sendJson(response, 200, envelope(await shoot(), body));
      } else if (body.action === "eat") {
        sendJson(response, 200, envelope(await eat(), body));
      } else if (body.action === "deliver") {
        sendJson(response, 200, envelope(await deliver(), body));
      } else {
        sendJson(response, 400, envelope({ status: "error", reason: "unknown_action" }, body));
      }
    } else {
      sendJson(response, 200, envelope({ status: "ok" }, body));
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
