import { Radio } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const streamFeature: FeatureDef = {
  id: "stream",
  nav: { label: "Live stream", icon: Radio, path: "/stream", section: "Serve", order: 75 },
  routes: [{ path: "stream", ...lazyPage(() => import("./StreamPage")) }],
};
