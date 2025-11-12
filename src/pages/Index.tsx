import { useState } from "react";
import { ImageUpload } from "@/components/ImageUpload";
import { ResultsGrid } from "@/components/ResultsGrid";
import { Card } from "@/components/ui/card";
import { toast } from "sonner";
import { Search } from "lucide-react";

interface Result {
  filename: string;
  similarity: number;
  url: string;
}

const Index = () => {
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<Result[]>([]);

  const handleSearch = async (file: File) => {
    setIsSearching(true);
    setResults([]);

    try {
      // TODO: Replace with actual API endpoint
      // const formData = new FormData();
      // formData.append('image', file);
      // const response = await fetch('YOUR_BACKEND_URL/search', {
      //   method: 'POST',
      //   body: formData,
      // });
      // const data = await response.json();
      // setResults(data.results);

      // Simulated search with mock data
      await new Promise((resolve) => setTimeout(resolve, 2000));
      
      const mockResults: Result[] = Array.from({ length: 8 }, (_, i) => ({
        filename: `image_${i + 1}.jpg`,
        similarity: Math.random() * 0.3 + 0.7, // 70-100% similarity
        url: `https://picsum.photos/400/400?random=${i}`,
      })).sort((a, b) => b.similarity - a.similarity);

      setResults(mockResults);
      toast.success("Search completed successfully!");
    } catch (error) {
      toast.error("Search failed. Please try again.");
      console.error("Search error:", error);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-hero">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-primary rounded-lg">
              <Search className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">
                CBIR System
              </h1>
              <p className="text-xs text-muted-foreground">
                Content-Based Image Retrieval
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
            Find Similar Images
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload an image and our AI-powered system will find visually similar
            images using deep learning feature extraction
          </p>
        </div>

        {/* Upload Section */}
        <ImageUpload onSearch={handleSearch} isSearching={isSearching} />

        {/* Loading State */}
        {isSearching && (
          <div className="flex flex-col items-center justify-center mt-12 animate-pulse">
            <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4" />
            <p className="text-muted-foreground">
              Analyzing image features...
            </p>
          </div>
        )}

        {/* Results */}
        <ResultsGrid results={results} />

        {/* Info Section */}
        {results.length === 0 && !isSearching && (
          <div className="mt-16 max-w-4xl mx-auto">
            <div className="grid md:grid-cols-3 gap-6">
              <Card className="p-6 text-center shadow-card hover:shadow-glow transition-all">
                <div className="w-12 h-12 bg-gradient-primary rounded-lg mx-auto mb-4 flex items-center justify-center">
                  <span className="text-2xl text-primary-foreground">1</span>
                </div>
                <h3 className="font-semibold text-foreground mb-2">
                  Upload Image
                </h3>
                <p className="text-sm text-muted-foreground">
                  Select or drag & drop any image to start searching
                </p>
              </Card>

              <Card className="p-6 text-center shadow-card hover:shadow-glow transition-all">
                <div className="w-12 h-12 bg-gradient-primary rounded-lg mx-auto mb-4 flex items-center justify-center">
                  <span className="text-2xl text-primary-foreground">2</span>
                </div>
                <h3 className="font-semibold text-foreground mb-2">
                  AI Analysis
                </h3>
                <p className="text-sm text-muted-foreground">
                  Deep learning extracts visual features from your image
                </p>
              </Card>

              <Card className="p-6 text-center shadow-card hover:shadow-glow transition-all">
                <div className="w-12 h-12 bg-gradient-primary rounded-lg mx-auto mb-4 flex items-center justify-center">
                  <span className="text-2xl text-primary-foreground">3</span>
                </div>
                <h3 className="font-semibold text-foreground mb-2">
                  Get Results
                </h3>
                <p className="text-sm text-muted-foreground">
                  View similar images ranked by visual similarity
                </p>
              </Card>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 mt-20 py-6">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          Powered by Convolutional Autoencoder & FAISS Vector Search
        </div>
      </footer>
    </div>
  );
};

export default Index;
