/** Query keys + shared queries used by more than one feature (one place: no stringly-typed cache keys). */
import { useQuery } from "@tanstack/react-query";
import { api } from "./client";
import type { GpuSnapshot, Job, RunSummary, ServicesStatus } from "./types";

export const keys = {
  runs: ["runs"] as const,
  run: (name: string) => ["runs", name] as const,
  series: (name: string) => ["runs", name, "series"] as const,
  artifacts: (name: string) => ["runs", name, "artifacts"] as const,
  report: (name: string) => ["runs", name, "report"] as const,
  jobs: ["jobs"] as const,
  profiles: ["profiles"] as const,
  profile: (name: string) => ["profiles", name] as const,
  services: ["services"] as const,
  gpu: ["gpu"] as const,
  compare: (runs: string[]) => ["compare", ...runs] as const,
  registryModels: ["registry", "models"] as const,
  registryRuns: ["registry", "runs"] as const,
  registryInfo: ["registry", "info"] as const,
  scoringHealth: ["scoring", "health"] as const,
  scoringModels: ["scoring", "models"] as const,
  samples: ["scoring", "samples"] as const,
  decisions: ["scoring", "decisions"] as const,
  thresholds: ["scoring", "thresholds"] as const,
  list: (name: string) => ["scoring", "lists", name] as const,
  changelog: ["scoring", "changelog"] as const,
  data: ["data"] as const,
  documents: ["docs", "documents"] as const,
  document: (name: string) => ["docs", "documents", name] as const,
  diagrams: ["docs", "diagrams"] as const,
};

export const useRuns = () =>
  useQuery({ queryKey: keys.runs, queryFn: () => api.get<RunSummary[]>("/runs"), refetchInterval: 5000 });

export const useJobs = () =>
  useQuery({ queryKey: keys.jobs, queryFn: () => api.get<Job[]>("/jobs"), refetchInterval: 4000 });

export const useServices = () =>
  useQuery({ queryKey: keys.services, queryFn: () => api.get<ServicesStatus>("/services"), refetchInterval: 8000 });

export const useGpu = (interval = 2000) =>
  useQuery({ queryKey: keys.gpu, queryFn: () => api.get<GpuSnapshot>("/system/gpu"), refetchInterval: interval });
