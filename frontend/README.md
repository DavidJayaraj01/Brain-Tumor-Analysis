# MediMind BTIS - Frontend

Modern React web application for the MediMind Brain Tumor Intelligence System.

## 🚀 Tech Stack

- **React 19** - Latest React with concurrent features
- **TypeScript 5.9** - Type-safe development
- **Vite 7** - Lightning-fast build tool
- **Tailwind CSS 4** - Utility-first styling
- **shadcn/ui** - Beautiful, accessible components
- **Lucide Icons** - Modern icon library

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/           # shadcn/ui components
│   │   ├── UploadPanel.tsx
│   │   ├── ResultPanel.tsx
│   │   ├── ChatInterface.tsx
│   │   ├── DetailedAnalysis.tsx
│   │   └── MedicalReport.tsx
│   ├── services/
│   │   └── api.ts        # API client
│   ├── lib/
│   │   └── utils.ts      # Utility functions
│   ├── styles/
│   │   └── app.css       # Custom styles
│   ├── App.tsx           # Main application
│   ├── main.tsx          # Entry point
│   └── index.css         # Global styles
├── public/               # Static assets
├── index.html            # HTML template
├── vite.config.ts        # Vite configuration
├── tailwind.config.cjs   # Tailwind configuration
├── tsconfig.json         # TypeScript config
└── package.json          # Dependencies
```

## 🛠️ Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 🔧 Configuration

### API Endpoint

Edit `src/services/api.ts` to change the backend URL:

```typescript
const API_BASE_URL = 'http://localhost:8000';
```

### Environment Variables

Create a `.env` file for environment-specific settings:

```env
VITE_API_URL=http://localhost:8000
```

## 🎨 Features

### Upload & Analysis
- Drag-and-drop MRI upload
- Patient information form (age, sex, symptoms)
- Real-time analysis progress

### Results Display
- **Overview Tab**: Classification probabilities, confidence scores
- **Analysis Tab**: Detailed multi-agent analysis breakdown
- **Report Tab**: Professional medical report
- **AI Chat Tab**: Interactive Q&A about the diagnosis

### UI Components
- Dark theme optimized for medical imaging
- Responsive design for desktop and tablet
- Accessible components (ARIA compliant)

## 🐳 Docker

Build and run with Docker:

```bash
# Build image
docker build -t medimind-frontend .

# Run container
docker run -p 80:80 medimind-frontend
```

## 📝 Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start dev server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |

## 🔗 Related

- [Backend API](../backend/README.md)
- [Main Documentation](../README.md)
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
