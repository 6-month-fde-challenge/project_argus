import { Server } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const systemFeature: FeatureDef = {
  id: "system",
  nav: { label: "System", icon: Server, path: "/system", section: "Operate", order: 35 },
  routes: [{ path: "system", ...lazyPage(() => import("./SystemPage")) }],
};
