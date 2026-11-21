import { Rocket } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const launchFeature: FeatureDef = {
  id: "launch",
  nav: { label: "Launch", icon: Rocket, path: "/launch", section: "Operate", order: 20 },
  routes: [{ path: "launch", ...lazyPage(() => import("./LaunchPage")) }],
};
