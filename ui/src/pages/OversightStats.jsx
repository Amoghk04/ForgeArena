import { useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, Legend,
} from 'recharts'
import Card from '../components/ui/Card'
import Spinner from '../components/ui/Spinner'
import Badge from '../components/ui/Badge'
import { usePolling } from '../hooks/usePolling'
import { getOversightStats } from '../api/client'

const NEON = '#00d4ff'
const GREEN = '#00ff88'
const ORANGE = '#ff9900'
const PURPLE = '#9955ff'
const RED = '#ff3366'

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="custom-tooltip">
      <p className="text-muted mb-1 text-[11px]">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }} className="text-[11px]">
          {p.name}: <b>{typeof p.value === 'number' ? (p.value > 1 ? p.value.toFixed(1) : p.value.toFixed(3)) : p.value}</b>
        </p>
      ))}
    </div>
  )
}

const DOMAIN_LABELS = {
  customer_support: 'Cust. Support',
  legal_summarisation: 'Legal',
  code_review: 'Code Review',
  product_recommendation: 'Product Rec.',
  mixed: 'Mixed',
}

const CORRUPTION_LABEL = {
  FACTUAL_OMISSION: 'Factual Omission',
  BIAS_INJECTION: 'Bias Injection',
  TEMPORAL_SHIFT: 'Temporal Shift',
  AUTHORITY_FABRICATION: 'Auth. Fabrication',
  INSTRUCTION_OVERRIDE: 'Instr. Override',
}

const CORRUPTION_COLORS = ['#ff3366', '#ff9900', '#ffee00', '#9955ff', '#00d4ff']

export default function OversightStats() {
  const fetcher = useCallback(() => getOversightStats(), [])
  const { data: stats, loading, error } = usePolling(fetcher, 5000)

  const domainCorrData = stats?.per_domain_correction
    ? Object.entries(stats.per_domain_correction).map(([d, v]) => ({
        domain: DOMAIN_LABELS[d] || d,
        correction: +(v * 100).toFixed(1),
      }))
    : []

  const corruptionData = stats?.per_corruption_detection
    ? Object.entries(stats.per_corruption_detection).map(([c, v], i) => ({
        subject: CORRUPTION_LABEL[c] || c,
        detection: +(v * 100).toFixed(1),
        explanation: +((stats.per_corruption_explanation?.[c] ?? 0) * 100).toFixed(1),
        color: CORRUPTION_COLORS[i % CORRUPTION_COLORS.length],
      }))
    : []

  const topMetrics = [
    { label: 'Total Episodes', value: stats?.total_episodes ?? 0, color: NEON },
    { label: 'Detection Accuracy', value: stats?.detection_accuracy != null ? `${(stats.detection_accuracy * 100).toFixed(1)}%` : 'n/a', color: GREEN },
    { label: 'Mean Composite Reward', value: stats?.mean_composite_reward != null ? stats.mean_composite_reward.toFixed(4) : 'n/a', color: ORANGE },
  ]

  return (
    <div className="p-6 space-y-5 animate-fade-in">
      {/* Top metrics */}
      <div className="grid grid-cols-3 gap-4">
        {topMetrics.map(({ label, value, color }) => (
          <Card key={label} className="p-4" glow>
            <div className="text-2xl font-mono font-bold" style={{ color }}>
              {loading && !stats ? <Spinner size="sm" /> : value}
            </div>
            <div className="text-xs text-muted mt-1">{label}</div>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-5">
        {/* Domain correction bars */}
        <Card className="p-5" glow>
          <h3 className="text-sm font-semibold text-secondary mb-4">Correction Score by Domain</h3>
          {loading && !stats ? (
            <div className="flex items-center justify-center h-52"><Spinner /></div>
          ) : domainCorrData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={domainCorrData} layout="vertical" margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1c1c3a" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} tick={{ fill: '#7080aa', fontSize: 10, fontFamily: 'JetBrains Mono' }} unit="%" />
                <YAxis dataKey="domain" type="category" tick={{ fill: '#7080aa', fontSize: 10, fontFamily: 'JetBrains Mono' }} width={90} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="correction" radius={[0, 3, 3, 0]} name="Correction %">
                  {domainCorrData.map((d, i) => {
                    const colors = [NEON, PURPLE, GREEN, ORANGE, '#ffee00']
                    return <Cell key={i} fill={colors[i % colors.length]} style={{ filter: `drop-shadow(0 0 4px ${colors[i % colors.length]}60)` }} />
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-52 text-muted text-sm">no episode data yet</div>
          )}
        </Card>

        {/* Corruption radar */}
        <Card className="p-5" glow>
          <h3 className="text-sm font-semibold text-secondary mb-4">Detection & Explanation per Corruption Type</h3>
          {loading && !stats ? (
            <div className="flex items-center justify-center h-52"><Spinner /></div>
          ) : corruptionData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <RadarChart data={corruptionData}>
                <PolarGrid stroke="#1c1c3a" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#7080aa', fontSize: 9, fontFamily: 'JetBrains Mono' }} />
                <Radar dataKey="detection" name="Detection %" stroke={NEON} fill={NEON} fillOpacity={0.15} strokeWidth={2} />
                <Radar dataKey="explanation" name="Explanation %" stroke={PURPLE} fill={PURPLE} fillOpacity={0.10} strokeWidth={2} />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'JetBrains Mono', color: '#7080aa' }} />
                <Tooltip content={<CustomTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-52 text-muted text-sm">no episode data yet</div>
          )}
        </Card>
      </div>

      {/* Per-corruption detail table */}
      {corruptionData.length > 0 && (
        <Card className="p-5">
          <h3 className="text-sm font-semibold text-secondary mb-4">Per-Corruption Breakdown</h3>
          <div className="space-y-3">
            {corruptionData.map((d, i) => (
              <div key={d.subject} className="grid grid-cols-3 gap-4 items-center">
                <div>
                  <Badge value={Object.keys(CORRUPTION_LABEL)[i]} type="corruption" />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-muted">detection</span>
                    <span className="text-neon">{d.detection.toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 bg-base rounded-full overflow-hidden">
                    <div className="h-full rounded-full progress-neon" style={{ width: `${d.detection}%` }} />
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-muted">explanation</span>
                    <span className="text-purple-neon">{d.explanation.toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 bg-base rounded-full overflow-hidden">
                    <div className="h-full rounded-full" style={{ width: `${d.explanation}%`, background: PURPLE, boxShadow: `0 0 6px ${PURPLE}80` }} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {error && (
        <div className="text-red-neon text-sm bg-red-neon/10 border border-red-neon/20 rounded-lg px-4 py-3">
          {error}
        </div>
      )}
    </div>
  )
}
