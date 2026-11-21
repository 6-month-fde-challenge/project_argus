import { Database } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const dataFeature: FeatureDef = {
  id: "data",
  nav: { label: "Dataset", icon: Database, path: "/data", section: "Knowledge", order: 90 },
  routes: [{ path: "data", ...lazyPage(() => import("./DataPage")) }],
};
