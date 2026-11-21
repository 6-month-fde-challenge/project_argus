import { FlaskConical } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const runsFeature: FeatureDef = {
  id: "runs",
  nav: { label: "Runs", icon: FlaskConical, path: "/runs", section: "Operate", order: 30 },
  routes: [
    { path: "runs", ...lazyPage(() => import("./pages/RunsPage")) },
    { path: "runs/:run", ...lazyPage(() => import("./pages/RunDetailPage")) },
  ],
};
