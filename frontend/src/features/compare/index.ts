import { GitCompare } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const compareFeature: FeatureDef = {
  id: "compare",
  nav: { label: "Compare", icon: GitCompare, path: "/compare", section: "Models", order: 50 },
  routes: [{ path: "compare", ...lazyPage(() => import("./ComparePage")) }],
};
