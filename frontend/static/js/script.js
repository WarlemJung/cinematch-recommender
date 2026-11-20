/**
 * CineMatch Frontend Application Controller
 *
 * Security Invariants:
 * - Zero usage of innerHTML for dynamic or external content.
 * - All text nodes are populated strictly via textContent.
 * - Poster URLs are validated against an explicit HTTPS whitelist regex with graceful DOM fallback.
 */

document.addEventListener('DOMContentLoaded', () => {
    const FAVORITES_STORAGE_KEY = 'cinematch_favorites_v2';
    const SAFE_POSTER_REGEX = /^https:\/\/image\.tmdb\.org\/t\/p\/w\d+\/[A-Za-z0-9_-]+\.(jpg|jpeg|png|webp)$/i;

    // DOM Elements
    const searchInput = document.getElementById('movieSearch');
    const searchBtn = document.getElementById('searchBtn');
    const clearSearchBtn = document.getElementById('clearSearchBtn');
    const autocompleteList = document.getElementById('autocomplete-list');
    const brandHomeLink = document.getElementById('brandHomeLink');

    const genreFilter = document.getElementById('genreFilter');
    const minYearFilter = document.getElementById('minYearFilter');
    const maxYearFilter = document.getElementById('maxYearFilter');
    const minRatingFilter = document.getElementById('minRatingFilter');
    const algorithmSelect = document.getElementById('algorithmSelect');
    const applyFiltersBtn = document.getElementById('applyFiltersBtn');
    const resetFiltersBtn = document.getElementById('resetFiltersBtn');

    const statusBanner = document.getElementById('statusBanner');

    const favoritesToggleBtn = document.getElementById('favoritesToggleBtn');
    const favoritesCountBadge = document.getElementById('favoritesCountBadge');
    const favoritesSection = document.getElementById('favorites-section');
    const favoritesGrid = document.getElementById('favorites-grid');
    const clearFavoritesBtn = document.getElementById('clearFavoritesBtn');
    const closeFavoritesBtn = document.getElementById('closeFavoritesBtn');

    const resultsSection = document.getElementById('results-section');
    const recommendationsGrid = document.getElementById('recommendations-grid');
    const searchedTitle = document.getElementById('searched-title');
    const recommendationMeta = document.getElementById('recommendation-meta');
    const closeResultsBtn = document.getElementById('closeResultsBtn');

    const trendingGrid = document.getElementById('trending-grid');
    const catalogSectionTitle = document.getElementById('catalog-section-title');
    const catalogSectionSubtitle = document.getElementById('catalog-section-subtitle');
    const catalogStatsPill = document.getElementById('catalog-stats-pill');

    const prevPageBtn = document.getElementById('prevPageBtn');
    const nextPageBtn = document.getElementById('nextPageBtn');
    const pageIndicator = document.getElementById('pageIndicator');

    const movieModal = document.getElementById('movieModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const modalBodyContent = document.getElementById('modalBodyContent');

    // Application State
    let currentPage = 1;
    let totalPages = 1;
    const pageSize = 12;
    let catalogMode = 'trending'; // 'trending' | 'search'
    let activeSelectedMovieId = null;
    let debounceTimer = null;

    // Initialize
    updateFavoritesUI();
    fetchGenres();
    loadCatalogPage(1);

    // Event Listeners
    searchBtn.addEventListener('click', () => {
        hideAutocomplete();
        const query = searchInput.value.trim();
        if (query) {
            fetchRecommendations(query);
        } else {
            catalogMode = 'trending';
            loadCatalogPage(1);
        }
    });

    searchInput.addEventListener('input', () => {
        const val = searchInput.value.trim();
        clearSearchBtn.classList.toggle('hidden', val.length === 0);
        clearTimeout(debounceTimer);
        if (val.length < 2) {
            hideAutocomplete();
            return;
        }
        debounceTimer = setTimeout(() => fetchAutocompleteSuggestions(val), 220);
    });

    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            hideAutocomplete();
            const query = searchInput.value.trim();
            if (query) {
                fetchRecommendations(query);
            }
        } else if (e.key === 'Escape') {
            hideAutocomplete();
        }
    });

    clearSearchBtn.addEventListener('click', () => {
        searchInput.value = '';
        clearSearchBtn.classList.add('hidden');
        hideAutocomplete();
        searchInput.focus();
    });

    brandHomeLink.addEventListener('click', (e) => {
        e.preventDefault();
        resetFilters();
        resultsSection.classList.add('hidden');
        catalogMode = 'trending';
        loadCatalogPage(1);
    });

    applyFiltersBtn.addEventListener('click', () => {
        currentPage = 1;
        if ( searchInput.value.trim() && !resultsSection.classList.contains('hidden') ) {
            const target = activeSelectedMovieId || searchInput.value.trim();
            fetchRecommendations(target);
        }
        loadCatalogPage(1);
    });

    resetFiltersBtn.addEventListener('click', () => {
        resetFilters();
        loadCatalogPage(1);
    });

    closeResultsBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        activeSelectedMovieId = null;
    });

    favoritesToggleBtn.addEventListener('click', () => {
        const isHidden = favoritesSection.classList.toggle('hidden');
        favoritesToggleBtn.setAttribute('aria-expanded', String(!isHidden));
        if (!isHidden) {
            renderFavoritesGrid();
        }
    });

    closeFavoritesBtn.addEventListener('click', () => {
        favoritesSection.classList.add('hidden');
        favoritesToggleBtn.setAttribute('aria-expanded', 'false');
    });

    clearFavoritesBtn.addEventListener('click', () => {
        localStorage.removeItem(FAVORITES_STORAGE_KEY);
        updateFavoritesUI();
        renderFavoritesGrid();
    });

    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            loadCatalogPage(currentPage - 1);
        }
    });

    nextPageBtn.addEventListener('click', () => {
        if (currentPage < totalPages) {
            loadCatalogPage(currentPage + 1);
        }
    });

    closeModalBtn.addEventListener('click', closeModal);
    movieModal.addEventListener('click', (e) => {
        if (e.target === movieModal) closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !movieModal.classList.contains('hidden')) {
            closeModal();
        }
    });
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-wrapper')) {
            hideAutocomplete();
        }
    });

    // Helper Functions
    function clearContainer(el) {
        while (el.firstChild) {
            el.removeChild(el.firstChild);
        }
    }

    function showLoader(container) {
        clearContainer(container);
        const loader = document.createElement('div');
        loader.className = 'loader';
        loader.setAttribute('role', 'status');
        loader.setAttribute('aria-label', 'Carregando...');
        container.appendChild(loader);
    }

    function showEmptyState(container, message) {
        clearContainer(container);
        const box = document.createElement('div');
        box.className = 'empty-state';
        const p = document.createElement('p');
        p.textContent = message;
        box.appendChild(p);
        container.appendChild(box);
    }

    function showBanner(message) {
        statusBanner.textContent = message;
        statusBanner.classList.remove('hidden');
    }

    function hideBanner() {
        statusBanner.textContent = '';
        statusBanner.classList.add('hidden');
    }

    function isValidPosterUrl(url) {
        return typeof url === 'string' && SAFE_POSTER_REGEX.test(url);
    }

    function getFilterQueryParams() {
        const params = new URLSearchParams();
        if (genreFilter.value) params.set('genre', genreFilter.value);
        if (minYearFilter.value) params.set('min_year', minYearFilter.value);
        if (maxYearFilter.value) params.set('max_year', maxYearFilter.value);
        if (minRatingFilter.value) params.set('min_rating', minRatingFilter.value);
        return params;
    }

    function resetFilters() {
        genreFilter.value = '';
        minYearFilter.value = '';
        maxYearFilter.value = '';
        minRatingFilter.value = '';
        hideBanner();
    }

    // Favorites persistence
    function getFavoritesMap() {
        try {
            const raw = localStorage.getItem(FAVORITES_STORAGE_KEY);
            if (!raw) return {};
            const parsed = JSON.parse(raw);
            return typeof parsed === 'object' && parsed !== null ? parsed : {};
        } catch {
            return {};
        }
    }

    function isFavorite(movieId) {
        const map = getFavoritesMap();
        return Boolean(map[String(movieId)]);
    }

    function toggleFavorite(movie) {
        if (!movie || !movie.movie_id) return;
        const map = getFavoritesMap();
        const key = String(movie.movie_id);
        if (map[key]) {
            delete map[key];
        } else {
            map[key] = {
                movie_id: movie.movie_id,
                title: movie.title,
                release_year: movie.release_year,
                vote_average: movie.vote_average,
                vote_count: movie.vote_count,
                genres: movie.genres || [],
                overview: movie.overview || '',
                poster_url: movie.poster_url || null,
            };
        }
        try {
            localStorage.setItem(FAVORITES_STORAGE_KEY, JSON.stringify(map));
        } catch {
            showBanner('Não foi possível salvar nos favoritos do navegador.');
        }
        updateFavoritesUI();
        if (!favoritesSection.classList.contains('hidden')) {
            renderFavoritesGrid();
        }
    }

    function updateFavoritesUI() {
        const count = Object.keys(getFavoritesMap()).length;
        favoritesCountBadge.textContent = String(count);
    }

    function renderFavoritesGrid() {
        const favs = Object.values(getFavoritesMap());
        if (favs.length === 0) {
            showEmptyState(favoritesGrid, 'Você ainda não adicionou filmes aos favoritos. Clique no ícone de coração em qualquer card.');
            return;
        }
        renderMovieCards(favs, favoritesGrid, false);
    }

    // API Calls
    async function fetchGenres() {
        try {
            const res = await fetch('/api/genres');
            if (!res.ok) return;
            const data = await res.json();
            const items = Array.isArray(data.items) ? data.items : [];
            items.forEach((g) => {
                const opt = document.createElement('option');
                opt.value = g.name;
                opt.textContent = `${g.name} (${g.count})`;
                genreFilter.appendChild(opt);
            });
        } catch (err) {
            console.error('Erro ao carregar gêneros:', err);
        }
    }

    async function fetchAutocompleteSuggestions(query) {
        try {
            const params = new URLSearchParams({ q: query, page: '1', page_size: '7' });
            const res = await fetch(`/api/movies/search?${params.toString()}`);
            if (!res.ok) return;
            const data = await res.json();
            renderAutocomplete(Array.isArray(data.items) ? data.items : []);
        } catch (err) {
            console.error('Erro no autocomplete:', err);
        }
    }

    function renderAutocomplete(items) {
        clearContainer(autocompleteList);
        if (items.length === 0) {
            hideAutocomplete();
            return;
        }

        items.forEach((movie) => {
            const item = document.createElement('div');
            item.className = 'autocomplete-item';
            item.setAttribute('role', 'option');
            item.tabIndex = 0;

            const titleSpan = document.createElement('span');
            titleSpan.className = 'autocomplete-title';
            titleSpan.textContent = movie.title;

            const metaDiv = document.createElement('div');
            metaDiv.className = 'autocomplete-meta';

            if (movie.release_year) {
                const yearSpan = document.createElement('span');
                yearSpan.textContent = String(movie.release_year);
                metaDiv.appendChild(yearSpan);
            }

            const ratingSpan = document.createElement('span');
            ratingSpan.textContent = `★ ${Number(movie.vote_average || 0).toFixed(1)}`;
            metaDiv.appendChild(ratingSpan);

            item.appendChild(titleSpan);
            item.appendChild(metaDiv);

            const selectMovie = () => {
                searchInput.value = movie.title;
                activeSelectedMovieId = movie.movie_id;
                hideAutocomplete();
                fetchRecommendations(movie.movie_id, movie.title);
            };

            item.addEventListener('click', selectMovie);
            item.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') selectMovie();
            });

            autocompleteList.appendChild(item);
        });

        autocompleteList.classList.remove('hidden');
        searchInput.setAttribute('aria-expanded', 'true');
    }

    function hideAutocomplete() {
        autocompleteList.classList.add('hidden');
        searchInput.setAttribute('aria-expanded', 'false');
    }

    async function loadCatalogPage(page) {
        hideBanner();
        showLoader(trendingGrid);

        const params = getFilterQueryParams();
        params.set('page', String(page));
        params.set('page_size', String(pageSize));

        const hasFilters = Boolean(
            genreFilter.value || minYearFilter.value || maxYearFilter.value || minRatingFilter.value
        );

        let endpoint = '/api/trending';
        if (catalogMode === 'search' || hasFilters) {
            params.set('top_n', '120');
        } else {
            params.set('top_n', '60');
        }

        try {
            const res = await fetch(`${endpoint}?${params.toString()}`);
            const data = await res.json();

            if (!res.ok) {
                showBanner(data.error || 'Erro ao carregar catálogo de filmes.');
                showEmptyState(trendingGrid, 'Não foi possível carregar os filmes com os parâmetros informados.');
                return;
            }

            const items = Array.isArray(data.items) ? data.items : (Array.isArray(data) ? data : []);
            currentPage = data.page || 1;
            totalPages = data.total_pages || 1;

            updatePaginationUI(currentPage, totalPages, data.total || items.length);

            if (hasFilters) {
                catalogSectionTitle.textContent = 'Catálogo Filtrado (Weighted Rating)';
                catalogSectionSubtitle.textContent = 'Resultados ordenados pela relevância demográfica aplicando seus filtros.';
            } else {
                catalogSectionTitle.textContent = 'Top Charts & Filmes em Alta';
                catalogSectionSubtitle.textContent = 'Classificados pela fórmula bayesiana Weighted Rating (IMDB) combinando nota média e volume de votos.';
            }

            if (items.length === 0) {
                showEmptyState(trendingGrid, 'Nenhum filme encontrado para os filtros selecionados.');
            } else {
                renderMovieCards(items, trendingGrid, false);
            }
        } catch (err) {
            console.error('Erro ao buscar catálogo:', err);
            showBanner('Falha de comunicação com o servidor.');
            showEmptyState(trendingGrid, 'Erro ao carregar catálogo.');
        }
    }

    function updatePaginationUI(page, totalPgs, totalItems) {
        pageIndicator.textContent = `Página ${page} de ${Math.max(1, totalPgs)}`;
        prevPageBtn.disabled = page <= 1;
        nextPageBtn.disabled = page >= totalPgs;
        catalogStatsPill.textContent = `${totalItems} filmes qualificados`;
    }

    async function fetchRecommendations(identifier, displayTitle = null) {
        hideBanner();
        showLoader(recommendationsGrid);
        resultsSection.classList.remove('hidden');

        const mode = algorithmSelect.value;
        const isNumericId = typeof identifier === 'number' || /^\d+$/.test(String(identifier));
        const params = getFilterQueryParams();
        params.set('top_n', '12');

        let url = '';
        if (mode === 'hybrid') {
            if (isNumericId) {
                url = `/api/recommendations/hybrid/${encodeURIComponent(identifier)}?${params.toString()}`;
            } else {
                params.set('title', String(identifier));
                url = `/api/recommendations/hybrid?${params.toString()}`;
            }
        } else {
            if (isNumericId) {
                url = `/api/recommend/${encodeURIComponent(identifier)}?${params.toString()}`;
            } else {
                params.set('title', String(identifier));
                url = `/api/recommend?${params.toString()}`;
            }
        }

        searchedTitle.textContent = displayTitle || String(identifier);
        recommendationMeta.textContent = 'Calculando vetores de similaridade sob demanda...';

        try {
            const res = await fetch(url);
            const data = await res.json();

            if (!res.ok || data.error) {
                recommendationMeta.textContent = '';
                showEmptyState(recommendationsGrid, data.error || 'Filme não encontrado no catálogo.');
                return;
            }

            const sourceMovie = data.source_movie || {};
            activeSelectedMovieId = sourceMovie.movie_id || null;
            if (sourceMovie.title) {
                const yearSuffix = sourceMovie.release_year ? ` (${sourceMovie.release_year})` : '';
                searchedTitle.textContent = `${sourceMovie.title}${yearSuffix}`;
            }

            if (mode === 'hybrid' && data.active_weights) {
                const w = data.active_weights;
                const collabStatus = data.collaborative_active ? 'Ativo' : 'Standby (Cold-Start Fallback)';
                recommendationMeta.textContent =
                    `Motor Híbrido — Pesos Ativos: Conteúdo ${(w.content * 100).toFixed(0)}% | ` +
                    `Demográfico ${(w.demographic * 100).toFixed(0)}% | ` +
                    `Colaborativo ${(w.collaborative * 100).toFixed(0)}% (${collabStatus})`;
            } else {
                recommendationMeta.textContent = 'Motor Baseado em Conteúdo — TF-IDF Esparso + Similaridade de Cosseno sob demanda.';
            }

            const items = Array.isArray(data.items) ? data.items : [];
            if (items.length === 0) {
                showEmptyState(recommendationsGrid, 'Nenhuma recomendação encontrada com os filtros atuais.');
            } else {
                renderMovieCards(items, recommendationsGrid, mode === 'hybrid');
            }
        } catch (err) {
            console.error('Erro ao buscar recomendações:', err);
            recommendationMeta.textContent = '';
            showEmptyState(recommendationsGrid, 'Ocorreu um erro ao processar as recomendações.');
        }
    }

    // Safe DOM Card Builder (Zero innerHTML)
    function createPosterElement(movie) {
        const wrapper = document.createElement('div');
        wrapper.className = 'poster-wrapper';

        const createFallback = () => {
            const fb = document.createElement('div');
            fb.className = 'poster-fallback';
            const icon = document.createElement('i');
            icon.className = 'fa-solid fa-film';
            icon.setAttribute('aria-hidden', 'true');
            const label = document.createElement('span');
            label.textContent = movie.title || 'CineMatch';
            fb.appendChild(icon);
            fb.appendChild(label);
            return fb;
        };

        if (isValidPosterUrl(movie.poster_url)) {
            const img = document.createElement('img');
            img.className = 'poster-img';
            img.src = movie.poster_url;
            img.alt = `Pôster de ${movie.title}`;
            img.loading = 'lazy';
            img.addEventListener('error', () => {
                if (img.parentNode === wrapper) {
                    wrapper.removeChild(img);
                    wrapper.appendChild(createFallback());
                }
            });
            wrapper.appendChild(img);
        } else {
            wrapper.appendChild(createFallback());
        }

        // Favorite toggle button
        const favBtn = document.createElement('button');
        favBtn.type = 'button';
        const favActive = isFavorite(movie.movie_id);
        favBtn.className = `card-fav-btn${favActive ? ' active' : ''}`;
        favBtn.setAttribute('aria-label', favActive ? 'Remover dos favoritos' : 'Adicionar aos favoritos');
        const heartIcon = document.createElement('i');
        heartIcon.className = favActive ? 'fa-solid fa-heart' : 'fa-regular fa-heart';
        favBtn.appendChild(heartIcon);

        favBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleFavorite(movie);
            const nowFav = isFavorite(movie.movie_id);
            favBtn.classList.toggle('active', nowFav);
            heartIcon.className = nowFav ? 'fa-solid fa-heart' : 'fa-regular fa-heart';
            favBtn.setAttribute('aria-label', nowFav ? 'Remover dos favoritos' : 'Adicionar aos favoritos');
        });
        wrapper.appendChild(favBtn);

        // Score badge if present
        const displayScore = movie.final_score ?? movie.similarity_score ?? movie.score;
        if (typeof displayScore === 'number') {
            const badge = document.createElement('div');
            badge.className = 'score-badge';
            if (movie.final_score !== undefined) {
                badge.textContent = `Match ${(movie.final_score * 100).toFixed(0)}%`;
            } else if (movie.similarity_score !== undefined) {
                badge.textContent = `Sim ${(movie.similarity_score * 100).toFixed(0)}%`;
            } else {
                badge.textContent = `WR ${movie.score.toFixed(2)}`;
            }
            wrapper.appendChild(badge);
        }

        return wrapper;
    }

    function renderMovieCards(movies, container, showHybridExplanation = false) {
        clearContainer(container);

        movies.forEach((movie) => {
            const card = document.createElement('article');
            card.className = 'movie-card';

            card.appendChild(createPosterElement(movie));

            const content = document.createElement('div');
            content.className = 'card-content';

            // Header row: Title + Release Year
            const headerRow = document.createElement('div');
            headerRow.className = 'card-header-row';

            const titleEl = document.createElement('h3');
            titleEl.className = 'movie-title';
            titleEl.textContent = movie.title || 'Sem título';
            headerRow.appendChild(titleEl);

            if (movie.release_year) {
                const yearEl = document.createElement('span');
                yearEl.className = 'movie-year';
                yearEl.textContent = String(movie.release_year);
                headerRow.appendChild(yearEl);
            }
            content.appendChild(headerRow);

            // Genre tags
            if (Array.isArray(movie.genres) && movie.genres.length > 0) {
                const genreRow = document.createElement('div');
                genreRow.className = 'genre-Row';
                movie.genres.slice(0, 3).forEach((gName) => {
                    const tag = document.createElement('span');
                    tag.className = 'genre-tag';
                    tag.textContent = gName;
                    genreRow.appendChild(tag);
                });
                content.appendChild(genreRow);
            }

            // Synopsis
            const overviewEl = document.createElement('p');
            overviewEl.className = 'movie-overview';
            const rawOverview = movie.overview || '';
            overviewEl.textContent = rawOverview.length > 115
                ? `${rawOverview.substring(0, 115)}...`
                : (rawOverview || 'Sinopse não disponível no catálogo.');
            content.appendChild(overviewEl);

            // Hybrid explanation & score breakdown
            if (showHybridExplanation && movie.explanation) {
                const expBox = document.createElement('div');
                expBox.className = 'explanation-box';
                const expText = document.createElement('p');
                expText.textContent = movie.explanation;
                expBox.appendChild(expText);

                if (movie.individual_scores) {
                    const breakdown = document.createElement('div');
                    breakdown.className = 'score-breakdown';

                    const cChip = document.createElement('span');
                    cChip.className = 'score-chip';
                    cChip.textContent = `Conteúdo: ${(Number(movie.individual_scores.content || 0) * 100).toFixed(0)}%`;
                    breakdown.appendChild(cChip);

                    const dChip = document.createElement('span');
                    dChip.className = 'score-chip';
                    dChip.textContent = `Demográfico: ${(Number(movie.individual_scores.demographic || 0) * 100).toFixed(0)}%`;
                    breakdown.appendChild(dChip);

                    if (movie.individual_scores.collaborative !== null && movie.individual_scores.collaborative !== undefined) {
                        const cfChip = document.createElement('span');
                        cfChip.className = 'score-chip';
                        cfChip.textContent = `Colab: ${(Number(movie.individual_scores.collaborative) * 100).toFixed(0)}%`;
                        breakdown.appendChild(cfChip);
                    }
                    expBox.appendChild(breakdown);
                }
                content.appendChild(expBox);
            }

            // Rating & Votes Meta
            const metaRow = document.createElement('div');
            metaRow.className = 'movie-meta';

            const ratingSpan = document.createElement('span');
            ratingSpan.className = 'rating';
            const starIcon = document.createElement('i');
            starIcon.className = 'fa-solid fa-star';
            starIcon.setAttribute('aria-hidden', 'true');
            const ratingText = document.createTextNode(` ${Number(movie.vote_average || 0).toFixed(1)}`);
            ratingSpan.appendChild(starIcon);
            ratingSpan.appendChild(ratingText);

            const votesSpan = document.createElement('span');
            const userIcon = document.createElement('i');
            userIcon.className = 'fa-solid fa-users';
            userIcon.setAttribute('aria-hidden', 'true');
            const votesText = document.createTextNode(` ${Number(movie.vote_count || 0).toLocaleString()}`);
            votesSpan.appendChild(userIcon);
            votesSpan.appendChild(votesText);

            metaRow.appendChild(ratingSpan);
            metaRow.appendChild(votesSpan);
            content.appendChild(metaRow);

            // Card Actions: Details & Similar
            const actionsRow = document.createElement('div');
            actionsRow.className = 'card-actions';

            if (movie.movie_id) {
                const detailsBtn = document.createElement('button');
                detailsBtn.type = 'button';
                detailsBtn.className = 'btn-card-action';
                detailsBtn.textContent = 'Detalhes';
                detailsBtn.addEventListener('click', () => openMovieDetails(movie.movie_id));

                const simBtn = document.createElement('button');
                simBtn.type = 'button';
                simBtn.className = 'btn-card-action';
                simBtn.textContent = 'Similares';
                simBtn.addEventListener('click', () => {
                    searchInput.value = movie.title;
                    fetchRecommendations(movie.movie_id, movie.title);
                    resultsSection.scrollIntoView({ behavior: 'smooth' });
                });

                actionsRow.appendChild(detailsBtn);
                actionsRow.appendChild(simBtn);
            }
            content.appendChild(actionsRow);

            card.appendChild(content);
            container.appendChild(card);
        });
    }

    // Movie Details Modal
    async function openMovieDetails(movieId) {
        clearContainer(modalBodyContent);
        movieModal.classList.remove('hidden');
        showLoader(modalBodyContent);

        try {
            const res = await fetch(`/api/movies/${encodeURIComponent(movieId)}`);
            const movie = await res.json();
            clearContainer(modalBodyContent);

            if (!res.ok || movie.error) {
                const errP = document.createElement('p');
                errP.textContent = movie.error || 'Não foi possível carregar os detalhes deste filme.';
                modalBodyContent.appendChild(errP);
                return;
            }

            const titleEl = document.createElement('h2');
            titleEl.id = 'modalMovieTitle';
            titleEl.textContent = movie.release_year
                ? `${movie.title} (${movie.release_year})`
                : movie.title;
            modalBodyContent.appendChild(titleEl);

            if (movie.tagline) {
                const taglineEl = document.createElement('p');
                taglineEl.className = 'modal-tagline';
                taglineEl.textContent = `"${movie.tagline}"`;
                modalBodyContent.appendChild(taglineEl);
            }

            if (Array.isArray(movie.genres) && movie.genres.length > 0) {
                const genreRow = document.createElement('div');
                genreRow.className = 'genre-Row';
                movie.genres.forEach((g) => {
                    const tag = document.createElement('span');
                    tag.className = 'genre-tag';
                    tag.textContent = g;
                    genreRow.appendChild(tag);
                });
                modalBodyContent.appendChild(genreRow);
            }

            const metaGrid = document.createElement('div');
            metaGrid.className = 'modal-meta-grid';

            const addMetaItem = (label, value) => {
                if (!value) return;
                const item = document.createElement('div');
                item.className = 'modal-meta-item';
                const lbl = document.createElement('span');
                lbl.className = 'modal-meta-label';
                lbl.textContent = label;
                const val = document.createElement('span');
                val.className = 'modal-meta-value';
                val.textContent = String(value);
                item.appendChild(lbl);
                item.appendChild(val);
                metaGrid.appendChild(item);
            };

            addMetaItem('Avaliação', `★ ${Number(movie.vote_average || 0).toFixed(2)} / 10`);
            addMetaItem('Total de Votos', Number(movie.vote_count || 0).toLocaleString());
            addMetaItem('Lançamento', movie.release_date || 'N/D');
            if (movie.runtime > 0) addMetaItem('Duração', `${movie.runtime} min`);
            addMetaItem('Idioma Original', (movie.original_language || 'en').toUpperCase());
            if (movie.collection_name) addMetaItem('Franquia / Coleção', movie.collection_name);

            modalBodyContent.appendChild(metaGrid);

            const synopsisEl = document.createElement('p');
            synopsisEl.style.marginTop = '1rem';
            synopsisEl.style.color = 'var(--text-primary)';
            synopsisEl.textContent = movie.overview || 'Sinopse indisponível.';
            modalBodyContent.appendChild(synopsisEl);

            if (Array.isArray(movie.production_companies) && movie.production_companies.length > 0) {
                const prodP = document.createElement('p');
                prodP.style.marginTop = '1rem';
                prodP.style.fontSize = '0.85rem';
                prodP.style.color = 'var(--text-secondary)';
                prodP.textContent = `Estúdios: ${movie.production_companies.join(', ')}`;
                modalBodyContent.appendChild(prodP);
            }

            const actionRow = document.createElement('div');
            actionRow.className = 'card-actions';
            actionRow.style.marginTop = '1.5rem';

            const recBtn = document.createElement('button');
            recBtn.type = 'button';
            recBtn.className = 'btn-apply';
            recBtn.textContent = 'Gerar Recomendações Híbridas para este Filme';
            recBtn.addEventListener('click', () => {
                closeModal();
                searchInput.value = movie.title;
                fetchRecommendations(movie.movie_id, movie.title);
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            });
            actionRow.appendChild(recBtn);
            modalBodyContent.appendChild(actionRow);
        } catch (err) {
            console.error('Erro ao carregar detalhes do filme:', err);
        }
    }

    function closeModal() {
        movieModal.classList.add('hidden');
    }
});
