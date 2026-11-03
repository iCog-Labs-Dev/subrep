package com.subrep;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.inventory.ItemStack;
import org.bukkit.plugin.java.JavaPlugin;

public final class StartStatePlugin extends JavaPlugin implements Listener {
    private static final String BOT_NAME = "subrep_bot";

    @Override
    public void onEnable() {
        getServer().getPluginManager().registerEvents(this, this);
    }

    @EventHandler
    public void onPlayerJoin(PlayerJoinEvent event) {
        Player player = event.getPlayer();
        if (!BOT_NAME.equals(player.getName())) {
            return;
        }
        Bukkit.getScheduler().runTask(this, () -> applyStartState(player));
    }

    private void applyStartState(Player player) {
        World world = Bukkit.getWorld("world");
        if (world == null) {
            getLogger().severe("world is not loaded; start state was not applied");
            return;
        }
        player.teleport(new Location(world, 40.5, 64, 0.5));
        player.setHealth(20.0);
        player.setFoodLevel(14);
        player.setSaturation(0.0f);
        player.setExhaustion(0.0f);
        player.getInventory().clear();
        player.getInventory().setArmorContents(new ItemStack[] {
            new ItemStack(Material.IRON_BOOTS),
            null,
            null,
            new ItemStack(Material.IRON_HELMET)
        });
        give(player, Material.EMERALD, 20);
        give(player, Material.TORCH, 4);
        give(player, Material.CHARCOAL, 2);
        give(player, Material.COAL, 4);
        give(player, Material.IRON_BLOCK, 4);
        give(player, Material.PUMPKIN, 1);
        give(player, Material.COOKED_BEEF, 3);
        give(player, Material.GOLDEN_APPLE, 1);
        getLogger().info("Applied deterministic start state to " + BOT_NAME);
    }

    private void give(Player player, Material material, int amount) {
        player.getInventory().addItem(new ItemStack(material, amount));
    }
}
