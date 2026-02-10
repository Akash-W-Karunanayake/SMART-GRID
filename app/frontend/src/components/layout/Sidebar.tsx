import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Shield,
  TrendingUp,
  AlertTriangle,
  BarChart3,
  Settings,
  Zap,
  Play,
} from 'lucide-react';
import { clsx } from 'clsx';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Simulation', href: '/simulation', icon: Play, badge: 'OpenDSS' },
  { name: 'Self-Healing', href: '/self-healing', icon: Shield, badge: 'MARL+GNN' },
  { name: 'Forecasting', href: '/forecasting', icon: TrendingUp, badge: 'ML' },
  { name: 'Diagnostics', href: '/diagnostics', icon: AlertTriangle, badge: 'CNN+GNN' },
  { name: 'Net Load', href: '/net-load', icon: BarChart3, badge: 'Transformer' },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export default function Sidebar() {
  return (
    <div className="flex flex-col w-64 bg-slate-800 border-r border-slate-700">
      {/* Logo */}
      <div className="flex items-center h-16 px-4 border-b border-slate-700">
        <Zap className="w-8 h-8 text-blue-500" />
        <div className="ml-3">
          <h1 className="text-lg font-bold text-white">Smart Grid AI</h1>
          <p className="text-xs text-slate-400">Research Framework</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              clsx(
                'flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-300 hover:bg-slate-700 hover:text-white'
              )
            }
          >
            <item.icon className="w-5 h-5 mr-3" />
            <span className="flex-1">{item.name}</span>
            {item.badge && (
              <span className="px-2 py-0.5 text-xs bg-slate-600 text-slate-300 rounded">
                {item.badge}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700">
        <div className="text-xs text-slate-500">
          <p>Project: 25-26J-092</p>
          <p>SLIIT Research</p>
        </div>
      </div>
    </div>
  );
}
