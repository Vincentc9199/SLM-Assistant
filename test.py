import pyglet

class TestEventDispatcher(pyglet.event.EventDispatcher):
    def test_event(self):
        print("Dispatching 'test_event'")
        self.dispatch_event('on_test_event')

TestEventDispatcher.register_event_type('on_test_event')
dispatcher = TestEventDispatcher()

@dispatcher.event
def on_test_event():
    print("Test event dispatched and handled")

def run_test():
    print("Starting pyglet test")
    pyglet.clock.schedule_once(lambda dt: dispatcher.test_event(), 0)
    pyglet.app.run()

if __name__ == '__main__':
    run_test()