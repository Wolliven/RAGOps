const express = require('express');
const router = express.Router();
const sqlite3 = require('sqlite3');

const API_BASE_URL = 'http://127.0.0.1:8000';
const db = new sqlite3.Database('ragops.db');

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

router.get('/', (req, res) => {
  res.redirect('/ragops/search');
});

router.get('/search', (req, res) => {
  res.render('ragops/search', {
    title: 'RAGOps Search',
    query: '',
    top_k: 3,
    results: null,
    error: null
  });
});

router.post('/search', async (req, res, next) => {
  const query = req.body.query;
  const top_k = Number(req.body.top_k || 3);

  if (!query || query.trim() === '') {
    res.render('ragops/search', {
      title: 'RAGOps Search',
      query: '',
      top_k: top_k,
      results: null,
      error: '質問を入力してください。'
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
        query: query,
        top_k: top_k
      })
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text);
    }

    const data = await response.json();

    db.run(
      'insert into search_history (query, top_k, created_at) values (?, ?, ?)',
      query,
      top_k,
      new Date().toISOString()
    );

    res.render('ragops/search', {
      title: 'RAGOps Search',
      query: query,
      top_k: top_k,
      results: data.results,
      error: null
    });
  } catch (err) {
    console.log(err);
    res.render('ragops/search', {
      title: 'RAGOps Search',
      query: query,
      top_k: top_k,
      results: null,
      error: '検索に失敗しました。FastAPIサーバーが起動しているか確認してください。'
    });
  }
});

router.get('/history', (req, res, next) => {
  db.all(
    'select * from search_history order by id desc',
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

module.exports = router;