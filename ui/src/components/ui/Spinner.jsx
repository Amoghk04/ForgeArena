export default function Spinner({ size = 'md', className = '' }) {
  const sizes = { sm: 'w-4 h-4 border-2', md: 'w-6 h-6 border-2', lg: 'w-10 h-10 border-2' }
  return (
    <div
      className={`rounded-full border-transparent animate-spin ${sizes[size]} ${className}`}
      style={{ borderTopColor: '#00d4ff', borderRightColor: 'rgba(0,212,255,0.3)' }}
    />
  )
}
