import { Scale } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const rulesFeature: FeatureDef = {
  id: "rules",
  nav: { label: "Rules & policy", icon: Scale, path: "/rules", section: "Serve", order: 80 },
  routes: [{ path: "rules", ...lazyPage(() => import("./RulesPage")) }],
};
