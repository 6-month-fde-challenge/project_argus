/** The feature registry. Adding a feature = create src/features/<name>/index.ts and add it here. */
import type { FeatureDef } from "./feature";
import { awsFeature } from "@/features/aws";
import { compareFeature } from "@/features/compare";
import { dataFeature } from "@/features/data";
import { docsFeature } from "@/features/docs";
import { launchFeature } from "@/features/launch";
import { opsFeature } from "@/features/ops";
import { overviewFeature } from "@/features/overview";
import { pipelineFeature } from "@/features/pipeline";
import { profilesFeature } from "@/features/profiles";
import { registryFeature } from "@/features/registry";
import { rulesFeature } from "@/features/rules";
import { runsFeature } from "@/features/runs";
import { scoringFeature } from "@/features/scoring";
import { settingsFeature } from "@/features/settings";
import { streamFeature } from "@/features/stream";
import { systemFeature } from "@/features/system";

export const FEATURES: FeatureDef[] = [
  overviewFeature, pipelineFeature, launchFeature, runsFeature, compareFeature,
  registryFeature, scoringFeature, streamFeature, rulesFeature, systemFeature, dataFeature, docsFeature,
  opsFeature, settingsFeature, profilesFeature, awsFeature,
];
