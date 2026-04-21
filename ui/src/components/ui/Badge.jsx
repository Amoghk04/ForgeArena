const DOMAIN_COLORS = {
  customer_support: 'text-neon bg-neon/10 border-neon/20',
  legal_summarisation: 'text-purple-neon bg-purple-neon/10 border-purple-neon/20',
  code_review: 'text-green-neon bg-green-neon/10 border-green-neon/20',
  product_recommendation: 'text-orange-neon bg-orange-neon/10 border-orange-neon/20',
  mixed: 'text-yellow-neon bg-yellow-neon/10 border-yellow-neon/20',
}

const TIER_COLORS = {
  learnable: 'text-green-neon bg-green-neon/10 border-green-neon/20',
  'too-easy': 'text-neon bg-neon/10 border-neon/20',
  'too-hard': 'text-red-neon bg-red-neon/10 border-red-neon/20',
  unestimated: 'text-muted bg-muted/10 border-muted/20',
}

const CORRUPTION_COLORS = {
  FACTUAL_OMISSION: 'text-red-neon bg-red-neon/10 border-red-neon/20',
  BIAS_INJECTION: 'text-orange-neon bg-orange-neon/10 border-orange-neon/20',
  TEMPORAL_SHIFT: 'text-yellow-neon bg-yellow-neon/10 border-yellow-neon/20',
  AUTHORITY_FABRICATION: 'text-purple-neon bg-purple-neon/10 border-purple-neon/20',
  INSTRUCTION_OVERRIDE: 'text-neon bg-neon/10 border-neon/20',
}

export default function Badge({ value, type = 'default', className = '' }) {
  let colorClass = 'text-muted bg-muted/10 border-muted/20'
  if (type === 'domain') colorClass = DOMAIN_COLORS[value] || colorClass
  else if (type === 'tier') colorClass = TIER_COLORS[value] || colorClass
  else if (type === 'corruption') colorClass = CORRUPTION_COLORS[value] || colorClass

  const label = value?.replace(/_/g, ' ')?.toLowerCase()

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium border ${colorClass} ${className}`}>
      {label}
    </span>
  )
}
