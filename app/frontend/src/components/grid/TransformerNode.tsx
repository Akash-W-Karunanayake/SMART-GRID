import { Handle, Position } from 'reactflow';
import { TransformerSvg } from './GridSvgIcons';

/**
 * Custom ReactFlow node rendered as the standard transformer symbol
 * (two overlapping circles). Placed as an intermediate node between
 * the primary and secondary buses it connects.
 */
export default function TransformerNode({ data }: { data: any }) {
  const loadingPercent: number = data.loadingPercent ?? 0;
  const status: 'normal' | 'low' | 'high' =
    loadingPercent > 80 ? 'high' : loadingPercent > 60 ? 'low' : 'normal';

  return (
    <div className="relative flex flex-col items-center">
      <Handle type="target" position={Position.Top} className="!bg-slate-400 !w-2 !h-2" />

      <div className="flex flex-col items-center" title={data.label}>
        <TransformerSvg size={32} status={status} />
        <div className="text-[9px] text-slate-300 mt-0.5 max-w-[80px] truncate text-center">
          {data.shortLabel}
        </div>
        {data.kva != null && (
          <div className="text-[8px] text-slate-400">
            {data.kva} kVA
          </div>
        )}
        {loadingPercent > 0 && (
          <div className={`text-[8px] font-medium ${
            loadingPercent > 80 ? 'text-red-400' : loadingPercent > 60 ? 'text-amber-400' : 'text-green-400'
          }`}>
            {loadingPercent.toFixed(1)}%
          </div>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-slate-400 !w-2 !h-2" />
    </div>
  );
}
