const assert = require("assert");
const fs = require("fs");
const http = require("http");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");

function request(method, url, token, body) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const payload = body === undefined ? "" : JSON.stringify(body);
    const request = http.request({
      hostname: parsed.hostname,
      port: parsed.port,
      path: parsed.pathname,
      method,
      headers: {
        "X-SubRep-Token": token,
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(payload),
      },
    }, (response) => {
      let data = "";
      response.on("data", (chunk) => { data += chunk; });
      response.on("end", () => resolve({
        statusCode: response.statusCode,
        body: data ? JSON.parse(data) : null,
      }));
    });
    request.on("error", reject);
    request.end(payload);
  });
}

async function main() {
  const mockDir = fs.mkdtempSync(path.join(os.tmpdir(), "subrep-bot-mock-"));
  fs.mkdirSync(path.join(mockDir, "mineflayer-pathfinder"), { recursive: true });
  fs.writeFileSync(path.join(mockDir, "mineflayer.js"), `
    const EventEmitter = require("events");
    const bot = new EventEmitter();
    bot.entity = { position: { x: 0, y: 64, z: 0 } };
    bot.health = 20;
    bot.food = 20;
    bot.inventory = { items: () => [] };
    bot.loadPlugin = () => {};
    bot.quit = () => bot.emit("end");
    exports.createBot = () => { setImmediate(() => bot.emit("spawn")); return bot; };
  `);
  fs.writeFileSync(path.join(mockDir, "mineflayer-pathfinder", "index.js"), `
    exports.pathfinder = () => {};
    exports.Movements = function() {};
    exports.goals = { GoalBlock: function() {} };
  `);
  fs.writeFileSync(path.join(mockDir, "preload.js"), `
    const Module = require("module");
    const original = Module._load;
    Module._load = function(request, parent, isMain) {
      if (request === "mineflayer") return require(${JSON.stringify(path.join(mockDir, "mineflayer.js"))});
      if (request === "mineflayer-pathfinder") return require(${JSON.stringify(path.join(mockDir, "mineflayer-pathfinder", "index.js"))});
      return original.apply(this, arguments);
    };
  `);
  const token = "protocol-test-token";
  const child = spawn(process.execPath, ["index.js"], {
    cwd: __dirname,
    env: {
      ...process.env,
      NODE_PATH: mockDir,
      NODE_OPTIONS: `--require ${path.join(mockDir, "preload.js")}`,
      SUBREP_BOT_TOKEN: token,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  try {
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("bot did not start")), 5000);
      child.stdout.on("data", (data) => {
        if (data.toString().includes("Sidecar listening")) {
          clearTimeout(timer);
          resolve();
        }
      });
      child.on("exit", (code) => reject(new Error(`bot exited: ${code}`)));
    });
    const base = "http://127.0.0.1:25566";
    const health = await request("GET", `${base}/health`, token);
    assert.strictEqual(health.statusCode, 200);
    assert.strictEqual(health.body.auth, "offline");
    const oversized = await request("POST", `${base}/act`, token, {
      action: "wait",
      padding: "x".repeat(1024 * 1024),
    });
    assert.strictEqual(oversized.statusCode, 413);
    await Promise.all(Array.from({ length: 10 }, () =>
      request("POST", `${base}/observe`, token, {})));
    const limited = await request("POST", `${base}/observe`, token, {});
    assert.strictEqual(limited.statusCode, 429);
    await new Promise((resolve) => setTimeout(resolve, 1100));
    const first = await request("POST", `${base}/act`, token, {
      action: "wait",
      message_id: "one",
      snapshot_id: "dusk_v1",
      world_tick: 12000,
    });
    assert.strictEqual(first.statusCode, 200);
    const replay = await request("POST", `${base}/act`, token, {
      action: "wait",
      message_id: "one",
    });
    assert.strictEqual(replay.statusCode, 409);
    await request("POST", `${base}/stop`, token, {});
    await new Promise((resolve) => child.once("exit", resolve));
    await assert.rejects(() => request("POST", `${base}/observe`, token, {}));
  } finally {
    if (!child.killed) child.kill("SIGTERM");
    fs.rmSync(mockDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
