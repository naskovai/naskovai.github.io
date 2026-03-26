(function () {
    function init() {
        var toc = document.querySelector('.toc.side');
        if (!toc) return;

        // Backdrop
        var backdrop = document.createElement('div');
        backdrop.className = 'toc-backdrop';
        document.body.appendChild(backdrop);

        // FAB button
        // Wrap scroll-to-top and FAB in a shared container
        var topLink = document.getElementById('top-link');
        var container = document.createElement('div');
        container.className = 'nav-fab-group';
        if (topLink) {
            topLink.parentNode.insertBefore(container, topLink);
            container.appendChild(topLink);
        } else {
            document.body.appendChild(container);
        }

        var fab = document.createElement('button');
        fab.className = 'toc-fab';
        fab.setAttribute('aria-label', 'Table of Contents');
        fab.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="16" y2="12"/><line x1="3" y1="18" x2="11" y2="18"/></svg>';
        container.insertBefore(fab, topLink || null);

        var isOpen = false;

        function open() {
            isOpen = true;
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

        // Close on link click (user navigated to a section)
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
