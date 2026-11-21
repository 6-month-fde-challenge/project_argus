import { ScanSearch } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const scoringFeature: FeatureDef = {
  id: "scoring",
  nav: { label: "Scoring", icon: ScanSearch, path: "/scoring", section: "Serve", order: 70 },
  routes: [{ path: "scoring", ...lazyPage(() => import("./ScoringPage")) }],
};
