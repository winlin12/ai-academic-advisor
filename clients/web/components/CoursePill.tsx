type Props = {
  code: string;
  name: string;
};

export default function CoursePill({ code, name }: Props) {
  return (
    <div className="rounded-xl border border-orange-900/10 bg-white/75 p-4 shadow-[0_8px_20px_rgba(120,53,15,0.08)]">
      <p className="font-semibold text-orange-700">{code}</p>
      <p className="text-sm text-stone-700">{name}</p>
    </div>
  );
}
