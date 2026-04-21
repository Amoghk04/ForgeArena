import { useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Cell, PieChart, Pie, Legend,
} from 'recharts'
import { RefreshCw, Zap, Database, TrendingUp, Clock } from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Spinner from '../components/ui/Spinner'
import { usePolling } from '../hooks/usePolling'
import { getForgeQueue, getForgeStats } from '../api/client'

const NEON = '#00d4ff'
const GREEN = '#00ff88'
const RED = '#ff3366'
const ORANGE = '#ff9900'

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="custom-tooltip">
      <p className="text-muted mb-1">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: <b>{p.value}</b></p>
      ))}
    </div>
  )
}

const QUEUE_ZONES = [
  { key: 'learnable_count', label: 'Learnable', color: GREEN, desc: '0.20 ≤ pass@k ≤ 0.85' },
  { key: 'too_easy_count', label: 'Too Easy', color: NEON, desc: 'pass@k > 0.85' },
  { key: 'too_hard_count', label: 'Too Hard', color: RED, desc: 'pass@k < 0.20' },
  { key: 'pending_estimation_count', label: 'Pending Estimation', color: ORANGE, desc: 'awaiting pass@k estimate' },
]

export default function ForgeQueue() {
  const qFetcher = useCallback(() => getForgeQueue(), [])
  const sFetcher = useCallback(() => getForgeStats(), [])

  const { data: queue, loading: qLoad, refresh: qRefresh } = usePolling(qFetcher, 4000)
  const { data: stats, loading: sLoad } = usePolling(sFetcher, 4000)

  const barData = QUEUE_ZONES.map(z => ({
    name: z.label,
    count: queue?.[z.key] ?? 0,
    color: z.color,
  }))

  const pieData = QUEUE_ZONES
    .map(z => ({ name: z.label, value: queue?.[z.key] ?? 0, color: z.color }))
    .filter(d => d.value > 0)

  const totalTasks = (queue?.seed_task_count ?? 0) + (queue?.generated_task_count ?? 0)

  return (
    <div className="p-6 space-y-5 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-primary">Forge Queue</h2>
          <p className="text-xs text-muted font-mono mt-0.5">Curriculum scheduler — adaptive task difficulty management</p>
        </div>
        <Button onClick={qRefresh} variant="secondary" size="sm">
          <RefreshCw size={13} /> refresh
        </Button>
      </div>

      {/* Zone cards */}
      <div className="grid grid-cols-4 gap-4">
        {QUEUE_ZONES.map(z => (
          <Card key={z.key} className="p-4" glow>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono text-muted">{z.label}</span>
              {queue?.replenishment_triggered && z.key === 'learnable_count' && (
                <span className="text-xs font-mono text-orange-neon flex items-center gap-1"><Zap size={10}/> replenishing</span>
              )}
            </div>
            <div className="text-3xl font-mono font-bold" style={{ color: z.color }}>
              {queue ? queue[z.key] ?? 0 : <Spinner size="sm" />}
            </div>
            <div className="text-xs text-muted mt-1">{z.desc}</div>
          </Card>
        ))}
      </div>

      {/* Source cards */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { icon: Database, label: 'Seed Tasks', key: 'seed_task_count', color: NEON },
          { icon: Zap, label: 'Generated Tasks', key: 'generated_task_count', color: GREEN },
          { icon: TrendingUp, label: 'Acceptance Rate', key: null, color: ORANGE },
          { icon: Clock, label: 'Total Episodes', key: null, color: '#7080aa' },
        ].map(({ icon: Icon, label, key, color }) => {
          let value
          if (key) value = queue?.[key] ?? (qLoad ? null : 0)
          else if (label === 'Acceptance Rate') value = stats?.generator_acceptance_rate != null ? `${(stats.generator_acceptance_rate * 100).toFixed(1)}%` : (sLoad ? null : 'n/a')
          else value = stats?.total_episodes ?? (sLoad ? null : 0)

          return (
            <Card key={label} className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <Icon size={13} style={{ color }} />
                <span className="text-xs text-muted">{label}</span>
              </div>
              <div className="text-2xl font-mono font-bold" style={{ color }}>
                {value ?? <Spinner size="sm" />}
              </div>
            </Card>
          )
        })}
      </div>

      <div className="grid grid-cols-2 gap-5">
        {/* Bar chart */}
        <Card className="p-5" glow>
          <h3 className="text-sm font-semibold text-secondary mb-4">Queue Counts by Zone</h3>
          {barData.some(d => d.count > 0) ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData} margin={{ left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1c1c3a" />
                <XAxis dataKey="name" tick={{ fill: '#7080aa', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                <YAxis tick={{ fill: '#7080aa', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {barData.map((d, i) => (
                    <Cell key={i} fill={d.color} style={{ filter: `drop-shadow(0 0 6px ${d.color}60)` }} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-52 text-muted text-sm">
              {qLoad ? <Spinner /> : 'no data'}
            </div>
          )}
        </Card>

        {/* Pie */}
        <Card className="p-5" glow>
          <h3 className="text-sm font-semibold text-secondary mb-4">Zone Distribution</h3>
          {pieData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={48} outerRadius={72} dataKey="value" paddingAngle={4}>
                    {pieData.map((d, i) => (
                      <Cell key={i} fill={d.color} style={{ filter: `drop-shadow(0 0 8px ${d.color}60)` }} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex flex-wrap justify-center gap-3 mt-2">
                {pieData.map(d => (
                  <span key={d.name} className="flex items-center gap-1 text-xs font-mono text-secondary">
                    <span className="w-2 h-2 rounded-full" style={{ background: d.color }} />
                    {d.name}: {d.value}
                  </span>
                ))}
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-52 text-muted text-sm">
              {qLoad ? <Spinner /> : 'no data'}
            </div>
          )}
        </Card>
      </div>

      {/* Learnable threshold reference */}
      <Card className="p-4">
        <h3 className="text-xs font-mono text-muted uppercase tracking-widest mb-3">Difficulty Thresholds</h3>
        <div className="relative h-6 bg-base rounded-full overflow-hidden border border-border">
          <div className="absolute inset-y-0 left-0 w-[20%] bg-red-neon/20 flex items-center justify-center">
            <span className="text-[9px] font-mono text-red-neon">too-hard</span>
          </div>
          <div className="absolute inset-y-0 left-[20%] right-[15%] bg-green-neon/15 flex items-center justify-center">
            <span className="text-[9px] font-mono text-green-neon">learnable zone (0.20 – 0.85)</span>
          </div>
          <div className="absolute inset-y-0 right-0 w-[15%] bg-neon/15 flex items-center justify-center">
            <span className="text-[9px] font-mono text-neon">too-easy</span>
          </div>
          {/* Markers */}
          <div className="absolute inset-y-0" style={{ left: '20%', width: 1, background: '#00ff88', boxShadow: '0 0 4px #00ff88' }} />
          <div className="absolute inset-y-0" style={{ left: '85%', width: 1, background: '#00ff88', boxShadow: '0 0 4px #00ff88' }} />
        </div>
        <div className="flex justify-between text-[9px] font-mono text-muted mt-1">
          <span>0.00</span><span>0.20</span><span className="ml-auto mr-4">0.85</span><span>1.00</span>
        </div>
      </Card>
    </div>
  )
}
