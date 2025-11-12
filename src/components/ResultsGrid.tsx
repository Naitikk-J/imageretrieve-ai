import { Card } from "@/components/ui/card";

interface Result {
  filename: string;
  similarity: number;
  url: string;
}

interface ResultsGridProps {
  results: Result[];
}

export const ResultsGrid = ({ results }: ResultsGridProps) => {
  if (results.length === 0) return null;

  return (
    <div className="w-full max-w-7xl mx-auto mt-12 animate-fade-in">
      <h2 className="text-2xl font-bold text-foreground mb-6">
        Similar Images Found
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {results.map((result, index) => (
          <Card
            key={index}
            className="overflow-hidden hover:shadow-glow transition-all duration-300 animate-slide-up"
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <div className="aspect-square relative bg-muted">
              <img
                src={result.url}
                alt={result.filename}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="p-4">
              <p className="text-sm font-medium text-foreground truncate mb-1">
                {result.filename}
              </p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Similarity</span>
                <span className="text-sm font-semibold text-primary">
                  {(result.similarity * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};
