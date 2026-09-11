import { HeartPulse } from 'lucide-react'

import { DocumentsPanel } from '@/components/app/DocumentsPanel'
import { HealthBadge } from '@/components/app/HealthBadge'
import { QueryPanel } from '@/components/app/QueryPanel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Toaster } from '@/components/ui/sonner'

function App() {
  return (
    <div className="min-h-svh bg-background text-foreground">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-4xl items-center justify-between gap-4 px-4 py-4">
          <div className="flex items-center gap-2">
            <HeartPulse className="size-5 text-primary" />
            <div>
              <h1 className="text-base font-semibold leading-none">CardioRAG</h1>
              <p className="text-xs text-muted-foreground">
                Cardiovascular MRI &amp; AI literature, grounded Q&amp;A
              </p>
            </div>
          </div>
          <HealthBadge />
        </div>
      </header>

      <main className="mx-auto flex max-w-4xl flex-col gap-6 px-4 py-6">
        <Alert>
          <AlertTitle>Research and educational tool</AlertTitle>
          <AlertDescription>
            Not a medical diagnostic system. Answers are grounded in a small, fixed corpus of
            papers and must never be treated as medical advice or a clinical recommendation.
          </AlertDescription>
        </Alert>

        <Tabs defaultValue="query">
          <TabsList>
            <TabsTrigger value="query">Ask a question</TabsTrigger>
            <TabsTrigger value="documents">Indexed documents</TabsTrigger>
          </TabsList>
          <TabsContent value="query" className="pt-4">
            <QueryPanel />
          </TabsContent>
          <TabsContent value="documents" className="pt-4">
            <DocumentsPanel />
          </TabsContent>
        </Tabs>
      </main>

      <Toaster />
    </div>
  )
}

export default App
