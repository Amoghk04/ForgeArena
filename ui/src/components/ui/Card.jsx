export default function Card({ children, className = '', glow = false, onClick }) {
  const base = 'bg-card border border-border rounded-xl transition-all duration-200'
  const hover = onClick ? 'cursor-pointer hover:border-neon-dim hover:bg-card-hover' : ''
  const glowClass = glow ? 'shadow-neon-card border-border-bright' : ''
  return (
    <div className={`${base} ${hover} ${glowClass} ${className}`} onClick={onClick}>
      {children}
    </div>
  )
}
