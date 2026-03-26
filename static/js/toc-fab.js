(function () {
    function init() {
        var toc = document.querySelector('.toc.side');
        if (!toc) return;

        // Backdrop
        var backdrop = document.createElement('div');
        backdrop.className = 'toc-backdrop';
        document.body.appendChild(backdrop);

        // Wrap scroll-to-top + FAB in a shared flex container
        var topLink = document.getElementById('top-link');
        var container = document.createElement('div');
        container.className = 'nav-fab-group';

        var fab = document.createElement('button');
        fab.className = 'toc-fab';
        fab.setAttribute('aria-label', 'Table of Contents');
        fab.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="16" y2="12"/><line x1="3" y1="18" x2="11" y2="18"/></svg>';

        if (topLink) {
            // Insert container where top-link was, then pull top-link into it
            topLink.parentNode.insertBefore(container, topLink);
            container.appendChild(fab);
            container.appendChild(topLink);
        } else {
            container.appendChild(fab);
            document.body.appendChild(container);
        }

        var isOpen = false;

        function positionOverlay() {
            var rect = container.getBoundingClientRect();
            toc.style.bottom = (window.innerHeight - rect.top + 8) + 'px';
            toc.style.right = (window.innerWidth - rect.right) + 'px';
        }

        function open() {
            isOpen = true;
            positionOverlay();
            toc.classList.add('toc-open');
            backdrop.classList.add('toc-backdrop-active');
            var details = toc.querySelector('details');
            if (details) details.open = true;
        }

        function close() {
            isOpen = false;
            toc.classList.remove('toc-open');
            backdrop.classList.remove('toc-backdrop-active');
        }

        fab.addEventListener('click', function () { isOpen ? close() : open(); });
        backdrop.addEventListener('click', close);

        toc.querySelectorAll('a').forEach(function (a) {
            a.addEventListener('click', close);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
