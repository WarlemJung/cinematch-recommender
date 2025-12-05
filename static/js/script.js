document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('movieSearch');
    const searchBtn = document.getElementById('searchBtn');
    const trendingGrid = document.getElementById('trending-grid');
    const recommendationsGrid = document.getElementById('recommendations-grid');
    const resultsSection = document.getElementById('results-section');
    const searchedTitle = document.getElementById('searched-title');
    const closeResultsBtn = document.querySelector('.close-results');

    // Fetch Trending Movies on Load
    fetchTrendingMovies();

    // Search Event Listeners
    searchBtn.addEventListener('click', handleSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSearch();
    });

    closeResultsBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        searchInput.value = '';
    });

    async function fetchTrendingMovies() {
        try {
            const response = await fetch('/api/trending');
            const data = await response.json();
            renderMovies(data, trendingGrid);
        } catch (error) {
            console.error('Error fetching trending movies:', error);
        }
    }

    async function handleSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        // Show loading state
        recommendationsGrid.innerHTML = '<div class="loader"></div>';
        resultsSection.classList.remove('hidden');
        searchedTitle.textContent = query;

        try {
            const response = await fetch(`/api/recommend?title=${encodeURIComponent(query)}`);
            const data = await response.json();

            if (data.error) {
                recommendationsGrid.innerHTML = `<p style="text-align:center; color: #ff4757;">${data.error}</p>`;
            } else {
                // The API returns a list of titles (strings)
                // We need to convert them to objects to reuse renderMovies or create a specific render function
                // Since we only get titles for content-based, we'll render simple cards
                renderSimpleMovies(data, recommendationsGrid);
            }
        } catch (error) {
            console.error('Error fetching recommendations:', error);
            recommendationsGrid.innerHTML = '<p style="text-align:center; color: #ff4757;">Something went wrong. Please try again.</p>';
        }
    }

    function renderMovies(movies, container) {
        container.innerHTML = '';
        movies.forEach(movie => {
            const card = document.createElement('div');
            card.className = 'movie-card';
            
            // Truncate overview
            const overview = movie.overview ? (movie.overview.substring(0, 100) + '...') : 'No description available.';

            card.innerHTML = `
                <div class="movie-title">${movie.title}</div>
                <p style="font-size: 0.8rem; color: #888;">${overview}</p>
                <div class="movie-meta">
                    <span class="rating"><i class="fa-solid fa-star"></i> ${movie.vote_average}</span>
                    <span><i class="fa-solid fa-users"></i> ${movie.vote_count}</span>
                </div>
            `;
            container.appendChild(card);
        });
    }

    function renderSimpleMovies(titles, container) {
        container.innerHTML = '';
        titles.forEach(title => {
            const card = document.createElement('div');
            card.className = 'movie-card';
            
            card.innerHTML = `
                <div class="movie-title">${title}</div>
                <div class="movie-meta">
                    <span style="color: var(--accent-color);">Recommended</span>
                </div>
            `;
            container.appendChild(card);
        });
    }
});
