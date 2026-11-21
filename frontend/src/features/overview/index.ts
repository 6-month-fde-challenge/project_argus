import { LayoutDashboard } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const overviewFeature: FeatureDef = {
  id: "overview",
  nav: { label: "Overview", icon: LayoutDashboard, path: "/", section: "Operate", order: 10, end: true },
  routes: [{ index: true, ...lazyPage(() => import("./OverviewPage")) }],
};
