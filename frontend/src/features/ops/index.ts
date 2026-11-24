import { SquareTerminal } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const opsFeature: FeatureDef = {
  id: "operations",
  nav: { label: "Operations", icon: SquareTerminal, path: "/operations", section: "Operate", order: 36 },
  routes: [{ path: "operations", ...lazyPage(() => import("./OperationsPage")) }],
};
