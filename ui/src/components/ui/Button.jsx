export default function Button({ children, onClick, disabled, variant = 'primary', size = 'md', className = '', type = 'button' }) {
  const base = 'inline-flex items-center justify-center gap-2 font-semibold rounded-lg transition-all duration-150 select-none'

  const sizes = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-2.5 text-sm',
  }

  const variants = {
    primary: disabled
      ? 'bg-neon-dark text-muted cursor-not-allowed'
      : 'bg-neon text-base hover:brightness-110 active:scale-95 shadow-neon-sm',
    secondary: disabled
      ? 'bg-card border border-border text-muted cursor-not-allowed'
      : 'bg-card border border-border-bright text-secondary hover:text-primary hover:border-neon-dim active:scale-95',
    danger: disabled
      ? 'bg-card border border-border text-muted cursor-not-allowed'
      : 'bg-red-neon/10 border border-red-neon/30 text-red-neon hover:bg-red-neon/20 active:scale-95',
    ghost: 'bg-transparent text-secondary hover:text-primary active:scale-95',
  }

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${sizes[size]} ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  )
}
