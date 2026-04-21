import { NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Sword,
  ListTodo,
  Flame,
  BarChart3,
  TrendingUp,
  FlaskConical,
  Zap,
} from 'lucide-react'

const NAV = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/arena', icon: Sword, label: 'Episode Arena' },
  { to: '/tasks', icon: ListTodo, label: 'Task Bank' },
  { to: '/forge', icon: Flame, label: 'Forge Queue' },
  { to: '/oversight', icon: BarChart3, label: 'Oversight Stats' },
  { to: '/difficulty', icon: TrendingUp, label: 'Difficulty Curve' },
  { to: '/grader', icon: FlaskConical, label: 'Grader' },
]

export default function Sidebar() {
  return (
    <aside className="w-56 shrink-0 flex flex-col bg-panel border-r border-border h-screen sticky top-0">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-border">
        <div className="flex items-center gap-2">
          <Zap size={18} className="text-neon" style={{ filter: 'drop-shadow(0 0 6px #00d4ff)' }} />
          <span className="font-mono font-bold text-base text-neon glow-text tracking-tight">
            FORGE<span className="text-primary opacity-60">+</span>ARENA
          </span>
        </div>
        <p className="text-xs text-muted mt-1 font-mono">oversight monitor</p>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 overflow-y-auto">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-5 py-2.5 text-sm transition-all group relative ${
                isActive
                  ? 'text-neon bg-neon-dark'
                  : 'text-secondary hover:text-primary hover:bg-card'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <span className="absolute left-0 top-0 bottom-0 w-0.5 bg-neon"
                    style={{ boxShadow: '0 0 8px #00d4ff' }} />
                )}
                <Icon size={15} className={isActive ? 'text-neon' : 'text-muted group-hover:text-secondary'} />
                <span className="font-medium tracking-wide">{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-border">
        <p className="text-xs text-muted font-mono">v1.0 · Qwen2.5</p>
        <p className="text-xs text-muted font-mono">1.5B + 7B</p>
      </div>
    </aside>
  )
}
