import { Workflow } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const pipelineFeature: FeatureDef = {
  id: "pipeline",
  nav: { label: "Pipeline", icon: Workflow, path: "/pipeline", section: "Models", order: 40 },
  routes: [{ path: "pipeline", ...lazyPage(() => import("./PipelinePage")) }],
};
