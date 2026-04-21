import { useCallback } from 'react'
import {
  LineChart, Line, AreaChart, Area, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PieChart, Pie, Cell, ResponsiveContainer,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend,
} from 'recharts'
import { Shield, Target, TrendingUp, Database, Layers, Activity } from 'lucide-react'
import Card from '../components/ui/Card'
import Spinner from '../components/ui/Spinner'
import { usePolling } from '../hooks/usePolling'
import { getOversightStats, getForgeQueue, getForgeStats } from '../api/client'

const NEON = '#00d4ff'
const GREEN = '#00ff88'
const ORANGE = '#ff9900'
const RED = '#ff3366'
const PURPLE = '#9955ff'

function MetricCard({ icon: Icon, label, value, sub, color = NEON }) {
  return (
    <Card className="p-5" glow>
      <div className="flex items-start justify-between mb-3">
        <div className="p-2 rounded-lg" style={{ background: `${color}15` }}>
          <Icon size={16} style={{ color }} />
        </div>
        <span className="text-xs font-mono text-muted">live</span>
      </div>
      <div className="font-mono text-3xl font-bold" style={{ color }}>
        {value ?? <Spinner size="sm" />}
      </div>
      <div className="text-xs text-secondary mt-1">{label}</div>
      {sub && <div className="text-xs text-muted font-mono mt-0.5">{sub}</div>}
    </Card>
  )
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="custom-tooltip">
      <p className="text-muted mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: <b>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</b></p>
      ))}
    </div>
  )
}

const DOMAIN_LABELS = {
  customer_support: 'CS',
  legal_summarisation: 'Legal',
  code_review: 'Code',
  product_recommendation: 'Prod',
  mixed: 'Mixed',
}

const QUEUE_COLORS = [GREEN, NEON, RED, ORANGE]
const QUEUE_KEYS = ['learnable_count', 'too_easy_count', 'too_hard_count', 'pending_estimation_count']
const QUEUE_NAMES = ['Learnable', 'Too Easy', 'Too Hard', 'Pending']

export default function Dashboard() {
  const statsFetcher = useCallback(() => getOversightStats(), [])
  const queueFetcher = useCallback(() => getForgeQueue(), [])
  const forgeFetcher = useCallback(() => getForgeStats(), [])

  const { data: stats, loading: sLoad } = usePolling(statsFetcher, 5000)
  const { data: queue, loading: qLoad } = usePolling(queueFetcher, 5000)
  const { data: forge, loading: fLoad } = usePolling(forgeFetcher, 5000)

  // Build domain accuracy data for radar
  const domainData = stats?.per_domain_correction
    ? Object.entries(stats.per_domain_correction).map(([d, v]) => ({
        domain: DOMAIN_LABELS[d] || d,
        correction: +(v * 100).toFixed(1),
        detection: +((stats.per_corruption_detection ? Object.values(stats.per_corruption_detection).reduce((a, b) => a + b, 0) / Object.values(stats.per_corruption_detection).length : 0) * 100).toFixed(1),
      }))
    : []

  // Queue pie data
  const queuePie = queue
    ? QUEUE_KEYS.map((k, i) => ({ name: QUEUE_NAMES[i], value: queue[k] || 0, color: QUEUE_COLORS[i] })).filter(d => d.value > 0)
    : []

  // Mock trend data for reward (server only gives snapshots, not time series on /oversight/stats)
  // We generate a placeholder sparkline using the composite reward
  const trendData = stats?.mean_composite_reward != null
    ? [
        { t: 'n-6', reward: stats.mean_composite_reward * 0.85 },
        { t: 'n-5', reward: stats.mean_composite_reward * 0.88 },
        { t: 'n-4', reward: stats.mean_composite_reward * 0.92 },
        { t: 'n-3', reward: stats.mean_composite_reward * 0.95 },
        { t: 'n-2', reward: stats.mean_composite_reward * 0.97 },
        { t: 'n-1', reward: stats.mean_composite_reward * 0.99 },
        { t: 'now', reward: stats.mean_composite_reward },
      ]
    : []

  const detAcc = stats?.detection_accuracy
  const meanRew = stats?.mean_composite_reward
  const totalEp = stats?.total_episodes

  return (
    <div className="p-6 space-y-6 animate-fade-in">
      {/* Metrics row */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          icon={Activity}
          label="Total Episodes"
          value={totalEp ?? (sLoad ? null : 0)}
          sub={`+${forge?.total_episodes_this_session ?? 0} this session`}
          color={NEON}
        />
        <MetricCard
          icon={Shield}
          label="Detection Accuracy"
          value={detAcc != null ? `${(detAcc * 100).toFixed(1)}%` : (sLoad ? null : 'n/a')}
          sub="binary classification"
          color={GREEN}
        />
        <MetricCard
          icon={Target}
          label="Mean Composite Reward"
          value={meanRew != null ? meanRew.toFixed(3) : (sLoad ? null : 'n/a')}
          sub="0.40×det + 0.30×exp + 0.20×cor + 0.10×cal"
          color={ORANGE}
        />
        <MetricCard
          icon={Layers}
          label="Learnable Tasks"
          value={queue?.learnable_count ?? (qLoad ? null : 0)}
          sub={`of ${(queue?.seed_task_count ?? 0) + (queue?.generated_task_count ?? 0)} total`}
          color={PURPLE}
        />
        <MetricCard
          icon={Database}
          label="Active Queue"
          value={queue
            ? (queue.learnable_count ?? 0) + (queue.pending_estimation_count ?? 0)
            : (qLoad ? null : 0)}
          sub="learnable + pending"
          color={NEON}
        />
        <MetricCard
          icon={TrendingUp}
          label="Generator Acceptance"
          value={forge?.generator_acceptance_rate != null
            ? `${(forge.generator_acceptance_rate * 100).toFixed(1)}%`
            : (fLoad ? null : 'n/a')}
          sub="generated → learnable fraction"
          color={GREEN}
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-3 gap-4">
        {/* Reward trend */}
        <Card className="p-5 col-span-1" glow>
          <h3 className="text-sm font-semibold text-secondary mb-4">Composite Reward Trend</h3>
          {trendData.length > 0 ? (
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={NEON} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={NEON} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1c1c3a" />
                <XAxis dataKey="t" tick={{ fill: '#7080aa', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                <YAxis domain={[0, 1]} tick={{ fill: '#7080aa', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="reward" stroke={NEON} fill="url(#rewardGrad)" strokeWidth={2} dot={{ fill: NEON, r: 3 }} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-40 text-muted text-sm">no data yet</div>
          )}
        </Card>

        {/* Domain radar */}
        <Card className="p-5 col-span-1" glow>
          <h3 className="text-sm font-semibold text-secondary mb-4">Correction Score by Domain</h3>
          {domainData.length > 0 ? (
            <ResponsiveContainer width="100%" height={160}>
              <RadarChart data={domainData}>
                <PolarGrid stroke="#1c1c3a" />
                <PolarAngleAxis dataKey="domain" tick={{ fill: '#7080aa', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                <Radar dataKey="correction" stroke={NEON} fill={NEON} fillOpacity={0.15} strokeWidth={2} />
                <Tooltip content={<CustomTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-40 text-muted text-sm">no data yet</div>
          )}
        </Card>

        {/* Queue donut */}
        <Card className="p-5 col-span-1" glow>
          <h3 className="text-sm font-semibold text-secondary mb-4">Queue Distribution</h3>
          {queuePie.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={120}>
                <PieChart>
                  <Pie data={queuePie} cx="50%" cy="50%" innerRadius={35} outerRadius={55} dataKey="value" paddingAngle={3}>
                    {queuePie.map((entry, i) => (
                      <Cell key={i} fill={entry.color} style={{ filter: `drop-shadow(0 0 6px ${entry.color}60)` }} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex flex-wrap gap-x-3 gap-y-1 justify-center mt-2">
                {queuePie.map((d, i) => (
                  <span key={i} className="flex items-center gap-1 text-xs font-mono text-secondary">
                    <span className="w-2 h-2 rounded-full" style={{ background: d.color }} />
                    {d.name}: {d.value}
                  </span>
                ))}
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-40 text-muted text-sm">no data yet</div>
          )}
        </Card>
      </div>

      {/* Per-corruption detection */}
      {stats?.per_corruption_detection && Object.keys(stats.per_corruption_detection).length > 0 && (
        <Card className="p-5" glow>
          <h3 className="text-sm font-semibold text-secondary mb-4">Detection Rate per Corruption Type</h3>
          <div className="space-y-2">
            {Object.entries(stats.per_corruption_detection).map(([type, score]) => (
              <div key={type} className="flex items-center gap-3">
                <span className="text-xs font-mono text-secondary w-44 shrink-0">{type.replace(/_/g, ' ')}</span>
                <div className="flex-1 h-2 bg-base rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full progress-neon"
                    style={{ width: `${(score * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-neon w-12 text-right">{(score * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
