import React from 'react';

export interface GridSvgProps {
  size?: number;
  color?: string;
  status?: 'normal' | 'low' | 'high' | 'off';
  className?: string;
}

const statusColors = {
  normal: '#22c55e',
  low: '#f59e0b',
  high: '#ef4444',
  off: '#6b7280',
};

function resolveColor(color?: string, status?: string): string {
  if (color) return color;
  return statusColors[(status as keyof typeof statusColors) ?? 'normal'] ?? statusColors.normal;
}

export function SolarPanelSvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Panel body */}
      <rect x="3" y="6" width="18" height="12" rx="1" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.2} />
      {/* Grid lines */}
      <line x1="9" y1="6" x2="9" y2="18" stroke={c} strokeWidth="1" />
      <line x1="15" y1="6" x2="15" y2="18" stroke={c} strokeWidth="1" />
      <line x1="3" y1="12" x2="21" y2="12" stroke={c} strokeWidth="1" />
      {/* Sun rays */}
      <circle cx="19" cy="4" r="2" fill={status === 'off' ? '#6b7280' : '#facc15'} />
      <line x1="19" y1="1" x2="19" y2="0" stroke={status === 'off' ? '#6b7280' : '#facc15'} strokeWidth="1" />
      <line x1="22" y1="4" x2="23" y2="4" stroke={status === 'off' ? '#6b7280' : '#facc15'} strokeWidth="1" />
      <line x1="21" y1="2" x2="22" y2="1" stroke={status === 'off' ? '#6b7280' : '#facc15'} strokeWidth="1" />
    </svg>
  );
}

export function TransformerSvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Two overlapping circles - standard transformer symbol */}
      <circle cx="10" cy="12" r="5" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.1} />
      <circle cx="14" cy="12" r="5" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.1} />
      {/* Connection lines */}
      <line x1="0" y1="12" x2="5" y2="12" stroke={c} strokeWidth="1.5" />
      <line x1="19" y1="12" x2="24" y2="12" stroke={c} strokeWidth="1.5" />
    </svg>
  );
}

export function SubstationSvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Building outline */}
      <rect x="4" y="4" width="16" height="16" rx="1" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.15} />
      {/* High voltage symbol */}
      <path d="M12 7L9 13h3l-1 5 5-7h-3.5L12 7z" fill={c} fillOpacity={0.8} stroke={c} strokeWidth="0.5" />
      {/* Base */}
      <line x1="2" y1="20" x2="22" y2="20" stroke={c} strokeWidth="1.5" />
    </svg>
  );
}

export function HouseSvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Roof */}
      <path d="M12 3L2 12h3v8h14v-8h3L12 3z" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.15} strokeLinejoin="round" />
      {/* Door */}
      <rect x="10" y="15" width="4" height="5" stroke={c} strokeWidth="1" fill={c} fillOpacity={0.3} />
      {/* Window */}
      <rect x="6" y="13" width="3" height="3" stroke={c} strokeWidth="1" fill={c} fillOpacity={0.3} />
    </svg>
  );
}

export function FactorySvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Building */}
      <path d="M2 20V10l6 4V10l6 4V8h8v12H2z" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.15} strokeLinejoin="round" />
      {/* Chimney */}
      <rect x="18" y="4" width="3" height="5" stroke={c} strokeWidth="1" fill={c} fillOpacity={0.3} />
      {/* Smoke */}
      <path d="M19.5 3c0-1 1-1.5 1-2.5" stroke={c} strokeWidth="0.8" strokeLinecap="round" opacity={0.5} />
    </svg>
  );
}

export function HospitalSvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Building */}
      <rect x="3" y="6" width="18" height="14" rx="1" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.15} />
      {/* Cross */}
      <rect x="10" y="9" width="4" height="8" rx="0.5" fill={c} fillOpacity={0.7} />
      <rect x="8" y="11" width="8" height="4" rx="0.5" fill={c} fillOpacity={0.7} />
      {/* Roof sign */}
      <rect x="8" y="4" width="8" height="3" rx="0.5" stroke={c} strokeWidth="1" fill={c} fillOpacity={0.3} />
    </svg>
  );
}

export function SchoolSvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Building */}
      <rect x="3" y="10" width="18" height="10" rx="1" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.15} />
      {/* Roof/pediment */}
      <path d="M2 10L12 4l10 6" stroke={c} strokeWidth="1.5" fill="none" strokeLinejoin="round" />
      {/* Flag pole */}
      <line x1="12" y1="4" x2="12" y2="1" stroke={c} strokeWidth="1" />
      <path d="M12 1h3v2h-3" fill={c} fillOpacity={0.5} />
      {/* Door */}
      <rect x="10" y="15" width="4" height="5" stroke={c} strokeWidth="1" fill={c} fillOpacity={0.3} />
    </svg>
  );
}

export function WindTurbineSvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Tower */}
      <line x1="12" y1="10" x2="12" y2="22" stroke={c} strokeWidth="1.5" />
      {/* Base */}
      <line x1="8" y1="22" x2="16" y2="22" stroke={c} strokeWidth="1.5" />
      {/* Hub */}
      <circle cx="12" cy="8" r="1.5" fill={c} />
      {/* Blades */}
      <path d="M12 8L12 1" stroke={c} strokeWidth="2" strokeLinecap="round" />
      <path d="M12 8L6.5 11.5" stroke={c} strokeWidth="2" strokeLinecap="round" />
      <path d="M12 8L17.5 11.5" stroke={c} strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function CommercialSvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Building */}
      <rect x="4" y="3" width="16" height="18" rx="1" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.15} />
      {/* Windows */}
      <rect x="7" y="6" width="3" height="3" stroke={c} strokeWidth="0.8" fill={c} fillOpacity={0.3} />
      <rect x="14" y="6" width="3" height="3" stroke={c} strokeWidth="0.8" fill={c} fillOpacity={0.3} />
      <rect x="7" y="11" width="3" height="3" stroke={c} strokeWidth="0.8" fill={c} fillOpacity={0.3} />
      <rect x="14" y="11" width="3" height="3" stroke={c} strokeWidth="0.8" fill={c} fillOpacity={0.3} />
      {/* Door */}
      <rect x="10" y="17" width="4" height="4" stroke={c} strokeWidth="0.8" fill={c} fillOpacity={0.3} />
    </svg>
  );
}

export function WaterPumpSvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Pump body */}
      <circle cx="12" cy="12" r="6" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.15} />
      {/* Water drops */}
      <path d="M12 8c0 0-2 2-2 3.5a2 2 0 004 0C14 10 12 8 12 8z" fill={c} fillOpacity={0.5} />
      {/* Pipes */}
      <line x1="2" y1="12" x2="6" y2="12" stroke={c} strokeWidth="2" />
      <line x1="18" y1="12" x2="22" y2="12" stroke={c} strokeWidth="2" />
    </svg>
  );
}

export function TelecomTowerSvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Tower structure */}
      <path d="M12 6L8 22h8L12 6z" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.1} strokeLinejoin="round" />
      {/* Antenna */}
      <line x1="12" y1="1" x2="12" y2="6" stroke={c} strokeWidth="1.5" />
      {/* Signal waves */}
      <path d="M8 4a5 5 0 018 0" stroke={c} strokeWidth="1" fill="none" opacity={0.6} />
      <path d="M6 2a8 8 0 0112 0" stroke={c} strokeWidth="1" fill="none" opacity={0.3} />
      {/* Cross bars */}
      <line x1="9.5" y1="12" x2="14.5" y2="12" stroke={c} strokeWidth="1" />
      <line x1="10" y1="17" x2="14" y2="17" stroke={c} strokeWidth="1" />
    </svg>
  );
}

export function BusNodeSvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Bus bar - horizontal thick line */}
      <rect x="3" y="10" width="18" height="4" rx="2" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.3} />
      {/* Connection stubs */}
      <line x1="7" y1="6" x2="7" y2="10" stroke={c} strokeWidth="1.5" />
      <line x1="12" y1="6" x2="12" y2="10" stroke={c} strokeWidth="1.5" />
      <line x1="17" y1="6" x2="17" y2="10" stroke={c} strokeWidth="1.5" />
      <line x1="7" y1="14" x2="7" y2="18" stroke={c} strokeWidth="1.5" />
      <line x1="12" y1="14" x2="12" y2="18" stroke={c} strokeWidth="1.5" />
      <line x1="17" y1="14" x2="17" y2="18" stroke={c} strokeWidth="1.5" />
    </svg>
  );
}

export function GeneratorSvg({ size = 24, color, status = 'normal', className }: GridSvgProps) {
  const c = resolveColor(color, status);
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      {/* Generator circle */}
      <circle cx="12" cy="12" r="8" stroke={c} strokeWidth="1.5" fill={c} fillOpacity={0.15} />
      {/* G letter */}
      <text x="12" y="16" textAnchor="middle" fill={c} fontSize="10" fontWeight="bold" fontFamily="sans-serif">G</text>
      {/* Connection */}
      <line x1="20" y1="12" x2="24" y2="12" stroke={c} strokeWidth="1.5" />
    </svg>
  );
}

// Map node name patterns to SVG components
export function getGridSvgIcon(nodeName: string): React.FC<GridSvgProps> {
  const name = nodeName.toLowerCase();

  if (name.includes('pv') || name.includes('solar')) return SolarPanelSvg;
  if (name.includes('wind')) return WindTurbineSvg;
  if (name.includes('hospital') || name.includes('health')) return HospitalSvg;
  if (name.includes('school') || name.includes('college') || name.includes('university')) return SchoolSvg;
  if (name.includes('factory') || name.includes('industrial') || name.includes('ind')) return FactorySvg;
  if (name.includes('commercial') || name.includes('shop') || name.includes('market') || name.includes('store')) return CommercialSvg;
  if (name.includes('pump') || name.includes('water')) return WaterPumpSvg;
  if (name.includes('telecom') || name.includes('tower') || name.includes('bts')) return TelecomTowerSvg;
  if (name.includes('sub') || name.includes('gss') || name.includes('33kv') || name.includes('11kv')) return SubstationSvg;
  if (name.includes('house') || name.includes('residential') || name.includes('home') || name.includes('lv')) return HouseSvg;
  if (name.includes('building') || name.includes('apartment')) return CommercialSvg;
  if (name.includes('gen') || name.includes('generator')) return GeneratorSvg;

  return BusNodeSvg;
}
