import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Shield,
  TrendingUp,
  AlertTriangle,
  BarChart3,
  Settings,
  Zap,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react';
import { clsx } from 'clsx';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Self-Healing', href: '/self-healing', icon: Shield, badge: 'MARL+GNN' },
  { name: 'Forecasting', href: '/forecasting', icon: TrendingUp, badge: 'ML' },
  { name: 'Diagnostics', href: '/diagnostics', icon: AlertTriangle, badge: 'CNN+GNN' },
  { name: 'Net Load', href: '/net-load', icon: BarChart3, badge: 'Transformer' },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export default function Sidebar() {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <div
      className={clsx(
        'flex flex-col bg-slate-800 border-r border-slate-700 transition-all duration-300',
        isOpen ? 'w-64' : 'w-20'
      )}
    >
      {/* Logo + Toggle */}
      <div
        className={clsx(
          'flex items-center h-16 border-b border-slate-700',
          isOpen ? 'px-4 justify-between' : 'px-3 justify-center'
        )}
      >
        <div className={clsx('flex items-center', !isOpen && 'justify-center')}>
          <Zap className="w-8 h-8 text-blue-500 shrink-0" />
          {isOpen && (
            <div className="ml-3">
              <h1 className="text-lg font-bold text-white">Smart Grid AI</h1>
              <p className="text-xs text-slate-400">Research Framework</p>
            </div>
          )}
        </div>

        {isOpen && (
          <button
            onClick={() => setIsOpen(false)}
            className="text-slate-400 hover:text-white transition-colors"
            title="Collapse sidebar"
          >
            <PanelLeftClose className="w-5 h-5" />
          </button>
        )}
      </div>

      {!isOpen && (
        <div className="flex justify-center py-3 border-b border-slate-700">
          <button
            onClick={() => setIsOpen(true)}
            className="text-slate-400 hover:text-white transition-colors"
            title="Expand sidebar"
          >
            <PanelLeftOpen className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            title={!isOpen ? item.name : undefined}
            className={({ isActive }) =>
              clsx(
                'flex items-center text-sm font-medium rounded-lg transition-colors',
                isOpen ? 'px-3 py-2' : 'px-0 py-3 justify-center',
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-300 hover:bg-slate-700 hover:text-white'
              )
            }
          >
            <item.icon className={clsx('w-5 h-5 shrink-0', isOpen ? 'mr-3' : '')} />

            {isOpen && <span className="flex-1">{item.name}</span>}

            {isOpen && item.badge && (
              <span className="px-2 py-0.5 text-xs bg-slate-600 text-slate-300 rounded">
                {item.badge}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700">
        {isOpen ? (
          <div className="text-xs text-slate-500">
            <p>Project: 25-26J-092</p>
            <p>SLIIT Research</p>
          </div>
        ) : (
          <div className="text-[10px] text-slate-500 text-center">
            <p>25-26J-092</p>
          </div>
        )}
      </div>
    </div>
  );
}