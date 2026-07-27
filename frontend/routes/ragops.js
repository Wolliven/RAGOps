const express = require('express');
const path = require('path');
const multer = require('multer');
const sqlite3 = require('sqlite3');

const router = express.Router();

const API_BASE_URL = 'http://127.0.0.1:8000';

// Put the database in the Express project root instead of depending
// on whichever folder the terminal happens to be using.
const dbPath = path.join(__dirname, '..', 'ragops.db');
const db = new sqlite3.Database(dbPath);

db.serialize(() => {
  db.run(`
    create table if not exists search_history (
      id integer primary key autoincrement,
      query text not null,
      top_k integer not null,
      created_at text not null
    )
  `);
});


// Store the uploaded file temporarily in memory.
// Express receives it, then immediately forwards it to FastAPI.
const upload = multer({
  storage: multer.memoryStorage(),

  // Maximum file size of 15 MB.
  limits: {
    fileSize: 15 * 1024 * 1024
  },

  fileFilter: (req, file, callback) => {
    const extension = path.extname(file.originalname).toLowerCase();
    const allowedExtensions = new Set(['.txt', '.md', '.pdf']);

    if (!allowedExtensions.has(extension)) {
      callback(
        new Error('Please select a TXT, Markdown, or PDF file.')
      );
      return;
    }

    callback(null, true);
  }
});


/*
 * This helper prevents us from repeating all the default values
 * every time search.ejs is rendered.
 */
function renderSearch(res, values = {}) {
  res.render('ragops/search', {
    title: 'RAGOps Search',
    query: '',
    top_k: 3,
    results: null,
    error: null,
    upload_message: null,
    upload_error: null,
    upload_data: null,
    ...values
  });
}


router.get('/', (req, res) => {
  res.redirect('/ragops/search');
});


router.get('/search', (req, res) => {
  renderSearch(res);
});


/*
 * Receive a file from the browser and forward it to FastAPI.
 */
router.post('/upload', (req, res) => {
  upload.single('file')(req, res, async (uploadError) => {
    if (uploadError) {
      renderSearch(res, {
        upload_error: uploadError.message
      });
      return;
    }

    if (!req.file) {
      renderSearch(res, {
        upload_error: 'Please select a file.'
      });
      return;
    }

    try {
      const formData = new FormData();

      const fileBlob = new Blob(
        [req.file.buffer],
        {
          type: req.file.mimetype || 'application/octet-stream'
        }
      );

      // "file" must match the parameter expected by FastAPI.
      formData.append(
        'file',
        fileBlob,
        req.file.originalname
      );

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
      });

      const responseText = await response.text();

      if (!response.ok) {
        throw new Error(responseText);
      }

      let data;

      try {
        data = JSON.parse(responseText);
      } catch {
        data = {
          message: responseText
        };
      }

      renderSearch(res, {
        upload_message:
          `${req.file.originalname} was uploaded and processed successfully.`,
        upload_data: data
      });
    } catch (err) {
      console.error(err);

      renderSearch(res, {
        upload_error:
          'The upload failed. Make sure the FastAPI server is running.'
      });
    }
  });
});


router.post('/search', async (req, res) => {
  const query = String(req.body.query || '').trim();

  const requestedTopK = Number(req.body.top_k);

  const top_k = Number.isInteger(requestedTopK)
    ? Math.min(10, Math.max(1, requestedTopK))
    : 3;

  if (query === '') {
    renderSearch(res, {
      query: '',
      top_k,
      error: 'Please enter a question.'
    });
    return;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query,
        top_k
      })
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text);
    }

    const data = await response.json();

    db.run(
      `
        insert into search_history (query, top_k, created_at)
        values (?, ?, ?)
      `,
      [query, top_k, new Date().toISOString()],
      (dbError) => {
        if (dbError) {
          console.error('Could not save search history:', dbError);
        }
      }
    );

    renderSearch(res, {
      query,
      top_k,
      results: data.results
    });
  } catch (err) {
    console.error(err);

    renderSearch(res, {
      query,
      top_k,
      error:
        'The search failed. Make sure the FastAPI server is running.'
    });
  }
});


router.get('/history', (req, res, next) => {
  db.all(
    `
      select *
      from search_history
      order by id desc
    `,
    (err, rows) => {
      if (err) {
        next(err);
        return;
      }

      res.render('ragops/history', {
        title: 'Search History',
        history: rows
      });
    }
  );
});

router.post('/history/delete', (req, res, next) => {
  db.run(
    'delete from search_history',
    (err) => {
      if (err) {
        next(err);
        return;
      }

      res.redirect('/ragops/history');
    }
  );
});

module.exports = router;