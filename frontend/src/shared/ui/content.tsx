/** Content renderers: markdown, JSON, code (CodeMirror, read-only or editable). */
import { json } from "@codemirror/lang-json";
import { yaml } from "@codemirror/lang-yaml";
import CodeMirror from "@uiw/react-codemirror";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/shared/lib/utils";

export const MarkdownView = ({ source, className }: { source: string; className?: string }) => (
  <div className={cn("prose-argus", className)}>
    <Markdown remarkPlugins={[remarkGfm]}>{source}</Markdown>
  </div>
);

export function CodeView({ value, language = "json", onChange, height = "auto", maxHeight = "520px", className }: {
  value: string; language?: "json" | "yaml" | "text"; onChange?: (v: string) => void; height?: string; maxHeight?: string; className?: string;
}) {
  const extensions = language === "json" ? [json()] : language === "yaml" ? [yaml()] : [];
  return (
    <div className={cn("overflow-hidden rounded-xl border border-line text-xs", className)}>
      <CodeMirror value={value} theme="dark" extensions={extensions} editable={!!onChange} readOnly={!onChange}
        onChange={onChange} height={height} maxHeight={maxHeight}
        basicSetup={{ lineNumbers: true, foldGutter: true, highlightActiveLine: !!onChange }} />
    </div>
  );
}

export const JsonView = ({ data, maxHeight }: { data: unknown; maxHeight?: string }) => (
  <CodeView value={JSON.stringify(data, null, 2)} language="json" maxHeight={maxHeight} />
);

/** Compact key/value grid for small objects (metrics, config sections). */
export function KeyValue({ data, className }: { data: Record<string, unknown>; className?: string }) {
  return (
    <dl className={cn("grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1.5 text-xs", className)}>
      {Object.entries(data).map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted truncate">{k.replaceAll("_", " ")}</dt>
          <dd className="text-right font-mono tabular-nums text-ink/90">
            {typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString("en-IN") : v.toFixed(4)) : v == null ? "–" : String(v)}
          </dd>
        </div>
      ))}
    </dl>
  );
}
