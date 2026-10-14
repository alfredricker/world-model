-- World-model probe (card 107). Mode "pasture" (default): builds a flat fenced pasture in front of the player and
-- spawns look-alike creatures in it. Mode "roam": puts the player on the ground at (wm_cx, wm_cz), the world as it
-- generates there. Either way it logs, every server step, the player's pose and each creature's position under a
-- fixed id (scoring only).
local N = tonumber(minetest.settings:get("wm_n") or "5")
local MOB = minetest.settings:get("wm_mob") or "mobs_mc:cow"
local SEED = tonumber(minetest.settings:get("wm_seed") or "1021")
local C = {x = 120, y = 16, z = 92}
local MODE = minetest.settings:get("wm_mode") or "pasture"
local RX, RZ = tonumber(minetest.settings:get("wm_cx") or "400"), tonumber(minetest.settings:get("wm_cz") or "400")
local rng = PcgRandom(SEED)
local f, ids, next_id, step, ready = nil, {}, 1, 0, false

local function build(player)
  for x = C.x - 12, C.x + 12 do
    for z = C.z - 2, C.z + 24 do
      minetest.set_node({x = x, y = C.y, z = z}, {name = "mcl_core:dirt_with_grass"})
      for y = C.y + 1, C.y + 12 do minetest.set_node({x = x, y = y, z = z}, {name = "air"}) end
      if x == C.x - 12 or x == C.x + 12 or z == C.z - 2 or z == C.z + 24 then
        minetest.set_node({x = x, y = C.y + 1, z = z}, {name = "mcl_fences:fence"})
      end
    end
  end
  player:set_pos({x = C.x, y = C.y + 0.5, z = C.z})
  player:set_look_horizontal(0)
  player:set_look_vertical(0.35)
  for i = 1, N do
    local p = {x = C.x - 6 + rng:next(0, 1200) / 100, y = C.y + 1, z = C.z + 5 + rng:next(0, 1000) / 100}
    minetest.add_entity(p, MOB)
  end
  ready = true
end

local function ground(x, z)
  -- the highest node at (x, z) a player stands on: walkable, not leaves, with no liquid above it
  for y = 90, -30, -1 do
    local name = minetest.get_node({x = x, y = y, z = z}).name
    local def = minetest.registered_nodes[name]
    if def and def.liquidtype ~= "none" and def.liquidtype ~= nil then return nil end
    if def and def.walkable and minetest.get_item_group(name, "leaves") == 0 then return y end
  end
  return nil
end

local function roam(player)
  for k = 0, 20 do                                      -- step east until dry ground
    local x = RX + 8 * k
    local y = ground(x, RZ)
    if y then
      player:set_pos({x = x, y = y + 0.5, z = RZ})
      player:set_look_horizontal(rng:next(0, 628) / 100)
      player:set_look_vertical(0.2)
      ready = true
      return
    end
  end
end

minetest.register_on_joinplayer(function(player)
  if MODE == "roam" then
    minetest.emerge_area({x = RX - 4, y = -30, z = RZ - 4}, {x = RX + 164, y = 90, z = RZ + 4},
      function(_, _, remaining) if remaining == 0 then roam(player) end end)
  else
    minetest.emerge_area({x = C.x - 14, y = C.y - 2, z = C.z - 4}, {x = C.x + 14, y = C.y + 14, z = C.z + 26},
      function(_, _, remaining) if remaining == 0 then build(player) end end)
  end
end)

minetest.register_globalstep(function(dtime)
  local player = minetest.get_connected_players()[1]
  if player == nil then return end
  if f == nil then f = io.open(minetest.get_worldpath() .. "/wm_probe.log", "w") end
  step = step + 1
  local pp = player:get_pos()
  local parts = {string.format("S %d %d %.5f %.4f %.4f %.4f %.5f %.5f", step, ready and 1 or 0, dtime, pp.x, pp.y, pp.z,
                               player:get_look_horizontal(), player:get_look_vertical())}
  for _, obj in ipairs(minetest.get_objects_inside_radius(pp, 40)) do
    local le = obj:get_luaentity()
    if le and le.name and string.find(le.name, "mobs_mc:") then
      if ids[le] == nil then ids[le] = next_id; next_id = next_id + 1 end
      local p = obj:get_pos()
      parts[#parts + 1] = string.format("M %d %s %.4f %.4f %.4f %.4f", ids[le], le.name, p.x, p.y, p.z, obj:get_yaw() or 0)
    end
  end
  f:write(table.concat(parts, " | ") .. "\n")
  f:flush()
end)

-- scoring only: the content id of every registered node, with its name and how it is drawn
minetest.register_on_mods_loaded(function()
  local g = io.open(minetest.get_worldpath() .. "/wm_nodes.txt", "w")
  for name, def in pairs(minetest.registered_nodes) do
    g:write(string.format("%d %s %s\n", minetest.get_content_id(name), name, def.drawtype or "normal"))
  end
  g:close()
end)
