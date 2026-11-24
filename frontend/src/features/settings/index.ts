import { SlidersHorizontal } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const settingsFeature: FeatureDef = {
  id: "settings",
  nav: { label: "Settings", icon: SlidersHorizontal, path: "/settings", section: "Operate", order: 38 },
  routes: [{ path: "settings", ...lazyPage(() => import("./SettingsPage")) }],
};
