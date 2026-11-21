import { BookOpen } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const docsFeature: FeatureDef = {
  id: "docs",
  nav: { label: "Docs & diagrams", icon: BookOpen, path: "/docs", section: "Knowledge", order: 100 },
  routes: [{ path: "docs", ...lazyPage(() => import("./DocsPage")) }],
};
